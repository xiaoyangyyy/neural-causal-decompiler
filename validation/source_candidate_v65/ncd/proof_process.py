"""Own-child Windows job limits, including venv launchers and descendants."""
import os,subprocess,time

def run_isolated(command,cwd,seconds,memory_bytes=8*1024**3):
    if os.name!='nt':raise ValueError('This resource backend requires Windows job objects')
    if not 0<seconds<=43200 or type(memory_bytes)is not int or memory_bytes<=0:raise ValueError('Worker resource limit')
    import ctypes
    from ctypes import wintypes as W
    class Basic(ctypes.Structure):
        _fields_=[('PerProcessUserTimeLimit',ctypes.c_longlong),('PerJobUserTimeLimit',ctypes.c_longlong),('LimitFlags',W.DWORD),('MinimumWorkingSetSize',ctypes.c_size_t),('MaximumWorkingSetSize',ctypes.c_size_t),('ActiveProcessLimit',W.DWORD),('Affinity',ctypes.c_size_t),('PriorityClass',W.DWORD),('SchedulingClass',W.DWORD)]
    class IO(ctypes.Structure):
        _fields_=[(name,ctypes.c_ulonglong) for name in ['ReadOperationCount','WriteOperationCount','OtherOperationCount','ReadTransferCount','WriteTransferCount','OtherTransferCount']]
    class Extended(ctypes.Structure):
        _fields_=[('BasicLimitInformation',Basic),('IoInfo',IO),('ProcessMemoryLimit',ctypes.c_size_t),('JobMemoryLimit',ctypes.c_size_t),('PeakProcessMemoryUsed',ctypes.c_size_t),('PeakJobMemoryUsed',ctypes.c_size_t)]
    class Accounting(ctypes.Structure):
        _fields_=[(n,ctypes.c_longlong) for n in ['TotalUserTime','TotalKernelTime','ThisPeriodTotalUserTime','ThisPeriodTotalKernelTime']]+[(n,W.DWORD) for n in ['TotalPageFaultCount','TotalProcesses','ActiveProcesses','TotalTerminatedProcesses']]
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.CreateJobObjectW.argtypes=[ctypes.c_void_p,W.LPCWSTR];kernel.CreateJobObjectW.restype=W.HANDLE
    kernel.SetInformationJobObject.argtypes=[W.HANDLE,ctypes.c_int,ctypes.c_void_p,W.DWORD];kernel.SetInformationJobObject.restype=W.BOOL
    kernel.AssignProcessToJobObject.argtypes=[W.HANDLE,W.HANDLE];kernel.AssignProcessToJobObject.restype=W.BOOL
    kernel.QueryInformationJobObject.argtypes=[W.HANDLE,ctypes.c_int,ctypes.c_void_p,W.DWORD,ctypes.c_void_p];kernel.QueryInformationJobObject.restype=W.BOOL
    kernel.OpenProcess.argtypes=[W.DWORD,W.BOOL,W.DWORD];kernel.OpenProcess.restype=W.HANDLE
    kernel.IsProcessInJob.argtypes=[W.HANDLE,W.HANDLE,ctypes.POINTER(W.BOOL)];kernel.IsProcessInJob.restype=W.BOOL
    kernel.WaitForSingleObject.argtypes=[W.HANDLE,W.DWORD];kernel.WaitForSingleObject.restype=W.DWORD
    native=ctypes.WinDLL('ntdll');native.NtSuspendProcess.argtypes=[W.HANDLE];native.NtSuspendProcess.restype=ctypes.c_long
    kernel.TerminateJobObject.argtypes=[W.HANDLE,W.UINT];kernel.TerminateJobObject.restype=W.BOOL
    kernel.CloseHandle.argtypes=[W.HANDLE];kernel.CloseHandle.restype=W.BOOL
    job=kernel.CreateJobObjectW(None,None)
    if not job:raise ctypes.WinError(ctypes.get_last_error())
    child=None
    assigned=False
    def accounting():
        result=Accounting()
        if not kernel.QueryInformationJobObject(job,1,ctypes.byref(result),ctypes.sizeof(result),None):
            raise ctypes.WinError(ctypes.get_last_error())
        return result
    def owned_ids():
        capacity=16
        while capacity<=16384:
            buffer=ctypes.create_string_buffer(8+ctypes.sizeof(ctypes.c_size_t)*capacity)
            if kernel.QueryInformationJobObject(job,3,buffer,len(buffer),None):
                count=ctypes.c_ulong.from_buffer(buffer,4).value
                return list((ctypes.c_size_t*count).from_buffer(buffer,8))
            error=ctypes.get_last_error()
            if error!=234:raise ctypes.WinError(error)
            capacity*=2
        raise RuntimeError('Owned process count exceeds the explicit 16384-handle cleanup budget')
    def terminate_and_drain():
        # ActiveProcesses can reach zero before process handles are signaled.
        # Suspend each owned process, repeatedly enumerate until none can spawn
        # unobserved descendants, then retain and wait on the actual handles.
        deadline=time.monotonic()+5
        handles={}
        try:
            while True:
                for pid in owned_ids():
                    if pid in handles:continue
                    handle=kernel.OpenProcess(0x100000|0x800|0x1000,False,pid)
                    if not handle:
                        error=ctypes.get_last_error()
                        if error==87:continue  # Process disappeared before opening.
                        raise ctypes.WinError(error)
                    member=W.BOOL()
                    if not kernel.IsProcessInJob(handle,job,ctypes.byref(member)):
                        kernel.CloseHandle(handle);raise ctypes.WinError(ctypes.get_last_error())
                    if not member.value:
                        kernel.CloseHandle(handle);continue  # Never touch a recycled external PID.
                    handles[pid]=handle
                    status=native.NtSuspendProcess(handle)
                    if status<0:
                        remaining=max(0,int((deadline-time.monotonic())*1000))
                        if kernel.WaitForSingleObject(handle,remaining)!=0:
                            raise RuntimeError('Cannot quiesce owned process before termination')
                if set(owned_ids())<=set(handles):break
                if time.monotonic()>=deadline:
                    raise TimeoutError('Cannot quiesce owned descendants within cleanup budget')
            if not kernel.TerminateJobObject(job,1):
                raise ctypes.WinError(ctypes.get_last_error())
            for handle in handles.values():
                remaining=max(0,int((deadline-time.monotonic())*1000))
                if kernel.WaitForSingleObject(handle,remaining)!=0:
                    raise TimeoutError('Owned process handle did not signal within cleanup budget')
            while accounting().ActiveProcesses:
                if time.monotonic()>=deadline:
                    raise TimeoutError('Owned descendants did not terminate within cleanup budget')
                time.sleep(.001)
        finally:
            for handle in handles.values():kernel.CloseHandle(handle)
    try:
        limits=Extended();limits.BasicLimitInformation.LimitFlags=0x2000|0x200;limits.JobMemoryLimit=memory_bytes
        if not kernel.SetInformationJobObject(job,9,ctypes.byref(limits),ctypes.sizeof(limits)):raise ctypes.WinError(ctypes.get_last_error())
        child=subprocess.Popen(command,cwd=cwd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,creationflags=0x08000000|0x4)
        if not kernel.AssignProcessToJobObject(job,W.HANDLE(int(child._handle))):raise ctypes.WinError(ctypes.get_last_error())
        assigned=True
        native=ctypes.WinDLL('ntdll');native.NtResumeProcess.argtypes=[W.HANDLE];native.NtResumeProcess.restype=ctypes.c_long
        if native.NtResumeProcess(W.HANDLE(int(child._handle)))<0:raise RuntimeError('Cannot resume suspended owned worker')
        timed_out=False
        try:stdout,stderr=child.communicate(timeout=seconds)
        except subprocess.TimeoutExpired:
            timed_out=True;terminate_and_drain();stdout,stderr=child.communicate(timeout=5)
        cleanup_required=not timed_out and bool(accounting().ActiveProcesses)
        if cleanup_required:terminate_and_drain()
        resources={'job_memory_limit_bytes':memory_bytes,'descendants_included':True,'timeout':timed_out,'cleanup_budget_seconds':5,'descendants_terminated':True,'orphan_cleanup_required':cleanup_required}
        if job:
            measured=Extended();account=Accounting()
            if not kernel.QueryInformationJobObject(job,9,ctypes.byref(measured),ctypes.sizeof(measured),None):raise ctypes.WinError(ctypes.get_last_error())
            if not kernel.QueryInformationJobObject(job,1,ctypes.byref(account),ctypes.sizeof(account),None):raise ctypes.WinError(ctypes.get_last_error())
            resources.update(peak_job_memory_bytes=measured.PeakJobMemoryUsed,total_processes=account.TotalProcesses,active_processes_on_return=account.ActiveProcesses)
            if account.ActiveProcesses:raise RuntimeError('Owned worker remained active')
        return {'exit_code':child.returncode,'stdout':stdout,'stderr':stderr,'resources':resources}
    finally:
        try:
            if job and assigned and accounting().ActiveProcesses:
                terminate_and_drain()
        finally:
            if job:kernel.CloseHandle(job)
            if child and child.poll() is None:child.kill();child.communicate(timeout=5)
