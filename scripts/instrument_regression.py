from pathlib import Path
p=Path('ncd/raw_program_trace.py');s=p.read_text(encoding='utf-8')
s=s.replace('''    def __init__(self,rule,*,trace_dependence=False):''','''    def __init__(self,rule,*,trace_dependence=False,trace_regression=False):''')
s=s.replace('''self.program=rule;self.trace_dependence=bool(trace_dependence);self.nodes''','''self.program=rule;self.trace_dependence=bool(trace_dependence);self.trace_regression=bool(trace_regression);self.nodes''')
s=s.replace('''physical=[];virtual=[];dep_keys={}''','''physical=[];virtual=[];dep_keys={};reg_keys={}''')
needle='''                    self.catalog[p]=meta;virtual.append(p);dep_keys[p]=_key(descriptors[role])
            for i,child in enumerate(node.args):visit(child,path+"/"+str(i))
'''
replacement='''                    self.catalog[p]=meta;virtual.append(p);dep_keys[p]=_key(descriptors[role])
            if self.trace_regression and node.op=="regress":
                predictors=node.args[1];dimensions=len(predictors.args) if predictors.op=="columns" else 1
                coefficients=1+4*dimensions+dimensions*(dimensions-1)//2
                descriptor={"response":node.args[0].to_dict(),"predictors":predictors.to_dict()}
                for fold in (0,1):
                    for role in [*(f"mean_{j}" for j in range(dimensions)),*(f"std_{j}" for j in range(dimensions)),*(f"beta_{j}" for j in range(coefficients))]:
                        p=f"{path}/internal/fold_{fold}/{role}";meta={"kind":"vector","semantic_kind":"scalar","op":"reg_"+role,
                            "reg_parent":path,"reg_fold":fold,"reg_role":role}
                        self.catalog[p]=meta;virtual.append(p);reg_keys[p]=_key({"internal":"regression","fold":fold,"role":role,**descriptor})
            for i,child in enumerate(node.args):visit(child,path+"/"+str(i))
'''
assert needle in s;s=s.replace(needle,replacement)
s=s.replace('''        self.groups=make_groups(grouped,"group/");self.dep_groups=make_groups(dep_grouped,"dep_group/");self._all_groups={**self.groups,**self.dep_groups}
''','''        reg_grouped={}
        for path in virtual:
            if path in reg_keys:reg_grouped.setdefault(reg_keys[path],[]).append(path)
        self.groups=make_groups(grouped,"group/");self.dep_groups=make_groups(dep_grouped,"dep_group/");self.reg_groups=make_groups(reg_grouped,"reg_group/")
        self._all_groups={**self.groups,**self.dep_groups,**self.reg_groups}
''')
s=s.replace('''        if self.trace_dependence:payload["dependence_groups"]=self.dep_groups
''','''        if self.trace_dependence:payload["dependence_groups"]=self.dep_groups
        if self.trace_regression:payload["regression_groups"]=self.reg_groups
''')
needle='''                if ma.get("dep_parent")==mb.get("dep_parent") and "dep_role" in ma and "dep_role" in mb:
                    if mb["dep_role"] in self._DEP_ANCESTORS.get(ma["dep_role"],set()) or ma["dep_role"] in self._DEP_ANCESTORS.get(mb["dep_role"],set()):return False
        return True
'''
replacement='''                if ma.get("dep_parent")==mb.get("dep_parent") and "dep_role" in ma and "dep_role" in mb:
                    if mb["dep_role"] in self._DEP_ANCESTORS.get(ma["dep_role"],set()) or ma["dep_role"] in self._DEP_ANCESTORS.get(mb["dep_role"],set()):return False
                if ma.get("reg_parent")==mb.get("reg_parent") and ma.get("reg_fold")==mb.get("reg_fold") and "reg_role" in ma and "reg_role" in mb:
                    ra,rb=ma["reg_role"],mb["reg_role"]
                    if (ra.startswith(("mean_","std_")) and rb.startswith("beta_")) or (rb.startswith(("mean_","std_")) and ra.startswith("beta_")):return False
        return True
'''
assert needle in s;s=s.replace(needle,replacement)
needle='''            return numerator/max(abs(denominator),1e-12)
        def evaluate(node,path,dataset,index):
'''
replacement='''            return numerator/max(abs(denominator),1e-12)
        def regress(values,path,index):
            y=np.asarray(values[0],dtype=float);x=np.asarray(values[1],dtype=float)
            if x.ndim==1:x=x[:,None]
            order=np.lexsort(tuple(x[:,i] for i in reversed(range(x.shape[1]))));prediction=np.empty(len(y))
            def basis(z):return np.column_stack([np.ones(len(z)),z,z*z,np.sin(z),np.tanh(z),
                *[z[:,i]*z[:,j] for i in range(z.shape[1]) for j in range(i)]])
            for fold in (0,1):
                hold,fit=order[fold::2],order[1-fold::2];natural_mean=x[fit].mean(0);natural_std=x[fit].std(0).clip(1e-6)
                mean=np.array([scalar(path+f"/internal/fold_{fold}/mean_{j}",natural_mean[j],index) for j in range(x.shape[1])])
                std=np.array([max(abs(scalar(path+f"/internal/fold_{fold}/std_{j}",natural_std[j],index)),1e-6) for j in range(x.shape[1])])
                a=basis(np.clip((x[fit]-mean)/std,-10,10));b=basis(np.clip((x[hold]-mean)/std,-10,10))
                penalty=np.eye(a.shape[1])*.01;penalty[0,0]=1e-8;natural_beta=np.linalg.solve(a.T@a+penalty,a.T@y[fit])
                beta=np.array([scalar(path+f"/internal/fold_{fold}/beta_{j}",natural_beta[j],index) for j in range(len(natural_beta))])
                prediction[hold]=b@beta
            return prediction
        def evaluate(node,path,dataset,index):
'''
assert needle in s;s=s.replace(needle,replacement)
s=s.replace('''                value=dep(values,path,index) if self.trace_dependence and node.op=="dep" else _apply(node,values,samples)''','''                if self.trace_dependence and node.op=="dep":value=dep(values,path,index)
                elif self.trace_regression and node.op=="regress":value=regress(values,path,index)
                else:value=_apply(node,values,samples)''')
s=s.replace('''def dependence_scalar_groups(executor):return [executor.address(path) for path in sorted(executor.dep_groups)]''','''def dependence_scalar_groups(executor):return [executor.address(path) for path in sorted(executor.dep_groups)]
def regression_scalar_groups(executor):return [executor.address(path) for path in sorted(executor.reg_groups)]''')
p.write_text(s,encoding='utf-8')
