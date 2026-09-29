"""Own-child Windows job limits, including venv launchers and descendants."""
import os,subprocess

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
    kernel.CloseHandle.argtypes=[W.HANDLE];kernel.CloseHandle.restype=W.BOOL
    job=kernel.CreateJobObjectW(None,None)
    if not job:raise ctypes.WinError(ctypes.get_last_error())
    child=None
    try:
        limits=Extended();limits.BasicLimitInformation.LimitFlags=0x2000|0x200;limits.JobMemoryLimit=memory_bytes
        if not kernel.SetInformationJobObject(job,9,ctypes.byref(limits),ctypes.sizeof(limits)):raise ctypes.WinError(ctypes.get_last_error())
        child=subprocess.Popen(command,cwd=cwd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,creationflags=0x08000000|0x4)
        if not kernel.AssignProcessToJobObject(job,W.HANDLE(int(child._handle))):raise ctypes.WinError(ctypes.get_last_error())
        native=ctypes.WinDLL('ntdll');native.NtResumeProcess.argtypes=[W.HANDLE];native.NtResumeProcess.restype=ctypes.c_long
        if native.NtResumeProcess(W.HANDLE(int(child._handle)))<0:raise RuntimeError('Cannot resume suspended owned worker')
        timed_out=False
        try:stdout,stderr=child.communicate(timeout=seconds)
        except subprocess.TimeoutExpired:
            timed_out=True;kernel.CloseHandle(job);job=None;stdout,stderr=child.communicate()
        resources={'job_memory_limit_bytes':memory_bytes,'descendants_included':True,'timeout':timed_out}
        if job:
            measured=Extended();account=Accounting()
            if not kernel.QueryInformationJobObject(job,9,ctypes.byref(measured),ctypes.sizeof(measured),None):raise ctypes.WinError(ctypes.get_last_error())
            if not kernel.QueryInformationJobObject(job,1,ctypes.byref(account),ctypes.sizeof(account),None):raise ctypes.WinError(ctypes.get_last_error())
            resources.update(peak_job_memory_bytes=measured.PeakJobMemoryUsed,total_processes=account.TotalProcesses)
        return {'exit_code':child.returncode,'stdout':stdout,'stderr':stderr,'resources':resources}
    finally:
        if job:kernel.CloseHandle(job)
        if child and child.poll() is None:child.kill();child.communicate()
