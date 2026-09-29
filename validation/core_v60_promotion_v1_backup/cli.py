"""CLI for causal discovery, program extraction and intervention audits."""
import argparse
import json
from pathlib import Path
from .experiment import Config,run
from .verify import verify
from .io import read_json

def main(argv=None):
    parser=argparse.ArgumentParser(description="Neural Causal Decompiler")
    sub=parser.add_subparsers(dest="command",required=True)
    for command,help_text in (("run","Legacy bivariate baseline"),("research","Composed IR, circuit guidance, active CEGIS and PySR"),
                              ("graphs","3/5/8-node graphs and neural-to-SCM recovery"),("run-all","Bivariate research and multivariate SCM workflow")):
        p=sub.add_parser(command,help=help_text)
        p.add_argument("--output",type=Path,required=True)
        p.add_argument("--quick",action="store_true",help="Small integration run")
        p.add_argument("--seed",type=int)
        if command!="run-all":
            p.add_argument("--epochs",type=int)
            p.add_argument("--config",type=Path)
    for command,help_text in (("verify","Replay legacy artifacts"),("verify-research","Replay expanded bivariate research"),
                              ("verify-graphs","Replay multivariate and SCM research"),("verify-all","Replay full workflow"),
                              ("inspect","Print saved summary")):
        p=sub.add_parser(command,help=help_text);p.add_argument("directory",type=Path)
    for command in ("joint", "numeric", "raw-numeric", "dependence-numeric", "regression-numeric", "oblique-numeric", "causal-guided", "conservative-guided", "ood-guided", "graph-decoder", "distilled-pc", "factorized-graph", "cost-sensitive-graph", "symmetric-skeleton", "graph-global", "quadratic-readout", "manifold-intervention", "collateral-intervention", "end-to-end-mechanism", "node-context-graph", "factorized-node-context", "antisymmetric-direction", "pair-consistent", "active-intervention-graph", "active-end-to-end", "mechanism-search", "guided", "relational"):
        p=sub.add_parser(command,help=f"Run {command} experiment")
        p.add_argument("--output",type=Path,required=True)
        p.add_argument("--quick",action="store_true",help="Small integration run")
        p.add_argument("--seed",type=int)
        if command in ("numeric", "raw-numeric", "dependence-numeric", "regression-numeric", "oblique-numeric", "causal-guided", "conservative-guided", "ood-guided", "graph-decoder", "distilled-pc", "factorized-graph", "cost-sensitive-graph", "symmetric-skeleton", "graph-global", "quadratic-readout", "manifold-intervention", "collateral-intervention", "end-to-end-mechanism", "active-end-to-end", "guided"):
            p.add_argument("--source",type=Path,required=True,help="Frozen joint experiment directory")
        p=sub.add_parser(f"verify-{command}",help=f"Independently replay {command} experiment")
        p.add_argument("directory",type=Path)
    p=sub.add_parser("certified-finite",help="Certified exact realization of finite neural transducers")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--quick",action="store_true",help="Three-case integration run")
    p.add_argument("--seed",type=int)
    p.add_argument("--cases",type=int)
    p.add_argument("--config",type=Path)
    p=sub.add_parser("verify-certified-finite",help="Verify and replay finite realization certificates")
    p.add_argument("directory",type=Path)
    p=sub.add_parser("certified-traffic",help="Train and certify quantized traffic latent dynamics")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--quick",action="store_true",help="One-model integration run")
    p.add_argument("--seed",type=int)
    p.add_argument("--config",type=Path)
    p=sub.add_parser("verify-certified-traffic",help="Replay traffic training and verify certificates")
    p.add_argument("directory",type=Path)
    p=sub.add_parser("certified-continuous",help="Bound continuous neural-state distinguishability")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--quick",action="store_true",help="Small deterministic certificate run")
    p.add_argument("--config",type=Path)
    p=sub.add_parser("verify-certified-continuous",help="Verify continuous separation certificates")
    p.add_argument("directory",type=Path)
    p=sub.add_parser("certified-continuous-scale",help="Scale continuous neural certification across dimensions and horizons")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--quick",action="store_true",help="Two-profile integration run")
    p.add_argument("--seed",type=int)
    p.add_argument("--config",type=Path)
    p=sub.add_parser("verify-certified-continuous-scale",help="Replay continuous certification scaling evidence")
    p.add_argument("directory",type=Path)
    p=sub.add_parser("certified-continuous-nonlinear",help="Certify phase-crossing nonlinear neural dynamics")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--quick",action="store_true",help="One-profile nonlinear integration run")
    p.add_argument("--seed",type=int)
    p.add_argument("--config",type=Path)
    p=sub.add_parser("verify-certified-continuous-nonlinear",help="Replay nonlinear phase-crossing certificates")
    p.add_argument("directory",type=Path)
    p=sub.add_parser("certified-continuous-regions",help="Certify continuous initial-state region pairs")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--quick",action="store_true",help="One-profile continuous-region integration run")
    p.add_argument("--seed",type=int)
    p.add_argument("--config",type=Path)
    p=sub.add_parser("verify-certified-continuous-regions",help="Replay continuous initial-region certificates")
    p.add_argument("directory",type=Path)
    p=sub.add_parser("certified-continuous-cover",help="Certify minimal finite-horizon covers of continuous state domains")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--quick",action="store_true",help="One-dimensional continuous-cover integration run")
    p.add_argument("--seed",type=int)
    p.add_argument("--config",type=Path)
    p=sub.add_parser("verify-certified-continuous-cover",help="Replay continuous behavioral-cover certificates")
    p.add_argument("directory",type=Path)
    p=sub.add_parser("certified-continuous-realization",help="Certify an infinite-horizon finite realization with continuous controls")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--quick",action="store_true",help="One-dimensional integration run")
    p.add_argument("--seed",type=int)
    p.add_argument("--config",type=Path)
    p=sub.add_parser("verify-certified-continuous-realization",help="Replay the closed-realization proof")
    p.add_argument("directory",type=Path)
    p=sub.add_parser("certified-generic-grid",help="Certify an inductive finite realization of a supplied ReLU system")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--system",type=Path,help="Serialized continuous ReLU system; defaults to the coupled nonlinear benchmark")
    p.add_argument("--config",type=Path)
    p=sub.add_parser("verify-certified-generic-grid",help="Independently replay the generic grid realization")
    p.add_argument("directory",type=Path)
    p=sub.add_parser("certified-transition-lower",help="Prove a transition-aware minimum-state lower bound")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--system",type=Path,help="Scalar affine ReLU system; defaults to the 1D continuous benchmark")
    p.add_argument("--config",type=Path)
    p=sub.add_parser("verify-certified-transition-lower",help="Replay exact-rational transition-overlap inequalities")
    p.add_argument("directory",type=Path)
    p=sub.add_parser("certified-multiswitch-lower",help="Prove a stronger multi-switch minimum-state lower bound")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--system",type=Path,help="Scalar affine ReLU system; defaults to the 1D continuous benchmark")
    p.add_argument("--config",type=Path)
    p=sub.add_parser("verify-certified-multiswitch-lower",help="Replay exact-rational multi-switch inequalities")
    p.add_argument("directory",type=Path)
    p=sub.add_parser("certified-shifted-realization",help="Certify a smaller nine-center infinite-horizon realization")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--quick",action="store_true",help="One-dimensional integration run")
    p.add_argument("--seed",type=int)
    p.add_argument("--config",type=Path)
    p=sub.add_parser("verify-certified-shifted-realization",help="Replay nine-center realization certificates")
    p.add_argument("directory",type=Path)
    p=sub.add_parser("prove",help="Generate scoped original-project proof certificates")
    p.add_argument("--config",type=Path,required=True)
    p.add_argument("--resume",action="store_true")
    p=sub.add_parser("verify-proof",help="Independently replay proof bundles")
    p.add_argument("directory",type=Path)
    p=sub.add_parser("audit-requirements",help="Audit original claims without promoting scoped proofs")
    p.add_argument("directory",type=Path)
    p.add_argument("--require-closed",action="store_true")
    args=parser.parse_args(argv)
    try:
        if args.command in ("prove","verify-proof","audit-requirements"):
            from .original_proof_workflow import prove,verify_proof,audit_requirements
            if args.command=="prove":result=prove(args.config,args.resume)
            elif args.command=="verify-proof":result=verify_proof(args.directory)
            else:result=audit_requirements(args.directory)
            print(json.dumps(result,ensure_ascii=False,indent=2))
            if args.command=="audit-requirements" and args.require_closed and not result["overall_objective_achieved"]:
                return 1
            return 0
        elif args.command=="certified-finite":
            from .certified_experiment import CertifiedFiniteConfig,run_certified_finite
            if args.config:
                if args.quick or args.seed is not None or args.cases is not None:
                    parser.error("--config cannot combine with --quick, --seed or --cases")
                config=CertifiedFiniteConfig(**read_json(args.config))
            else:
                config=CertifiedFiniteConfig.quick() if args.quick else CertifiedFiniteConfig()
                if args.seed is not None:config.seed=args.seed
                if args.cases is not None:config.cases=args.cases
            print(json.dumps(run_certified_finite(args.output,config),ensure_ascii=False,indent=2))
        elif args.command=="verify-certified-finite":
            from .certified_experiment import verify_certified_finite
            print(json.dumps(verify_certified_finite(args.directory),ensure_ascii=False,indent=2))
        elif args.command=="certified-traffic":
            from .traffic_realization import TrafficConfig,run_certified_traffic
            if args.config:
                if args.quick or args.seed is not None:
                    parser.error("--config cannot combine with --quick or --seed")
                config=TrafficConfig(**read_json(args.config))
            else:
                config=TrafficConfig.quick() if args.quick else TrafficConfig()
                if args.seed is not None:config.seed=args.seed
            print(json.dumps(run_certified_traffic(args.output,config),ensure_ascii=False,indent=2))
        elif args.command=="verify-certified-traffic":
            from .traffic_realization import verify_certified_traffic
            print(json.dumps(verify_certified_traffic(args.directory),ensure_ascii=False,indent=2))
        elif args.command=="certified-continuous":
            from .continuous_separation import ContinuousSeparationConfig,run_continuous_separation
            if args.config:
                if args.quick:
                    parser.error("--config cannot combine with --quick")
                config=ContinuousSeparationConfig(**read_json(args.config))
            else:
                config=ContinuousSeparationConfig.quick() if args.quick else ContinuousSeparationConfig()
            print(json.dumps(run_continuous_separation(args.output,config),ensure_ascii=False,indent=2))
        elif args.command=="verify-certified-continuous":
            from .continuous_separation import verify_continuous_separation
            print(json.dumps(verify_continuous_separation(args.directory),ensure_ascii=False,indent=2))
        elif args.command=="certified-continuous-scale":
            from .continuous_scale import ContinuousScaleConfig,run_continuous_scale
            if args.config:
                if args.quick or args.seed is not None:
                    parser.error("--config cannot combine with --quick or --seed")
                config=ContinuousScaleConfig(**read_json(args.config))
            else:
                config=ContinuousScaleConfig.quick() if args.quick else ContinuousScaleConfig()
                if args.seed is not None:config.seed=args.seed
            print(json.dumps(run_continuous_scale(args.output,config),ensure_ascii=False,indent=2))
        elif args.command=="verify-certified-continuous-scale":
            from .continuous_scale import verify_continuous_scale
            print(json.dumps(verify_continuous_scale(args.directory),ensure_ascii=False,indent=2))
        elif args.command=="certified-continuous-nonlinear":
            from .continuous_nonlinear import NonlinearContinuousConfig,run_continuous_nonlinear
            if args.config:
                if args.quick or args.seed is not None:
                    parser.error("--config cannot combine with --quick or --seed")
                config=NonlinearContinuousConfig(**read_json(args.config))
            else:
                config=NonlinearContinuousConfig.quick() if args.quick else NonlinearContinuousConfig()
                if args.seed is not None:config.seed=args.seed
            print(json.dumps(run_continuous_nonlinear(args.output,config),ensure_ascii=False,indent=2))
        elif args.command=="verify-certified-continuous-nonlinear":
            from .continuous_nonlinear import verify_continuous_nonlinear
            print(json.dumps(verify_continuous_nonlinear(args.directory),ensure_ascii=False,indent=2))
        elif args.command=="certified-continuous-regions":
            from .continuous_regions import ContinuousRegionConfig,run_continuous_regions
            if args.config:
                if args.quick or args.seed is not None:
                    parser.error("--config cannot combine with --quick or --seed")
                config=ContinuousRegionConfig(**read_json(args.config))
            else:
                config=ContinuousRegionConfig.quick() if args.quick else ContinuousRegionConfig()
                if args.seed is not None:config.seed=args.seed
            print(json.dumps(run_continuous_regions(args.output,config),ensure_ascii=False,indent=2))
        elif args.command=="verify-certified-continuous-regions":
            from .continuous_regions import verify_continuous_regions
            print(json.dumps(verify_continuous_regions(args.directory),ensure_ascii=False,indent=2))
        elif args.command=="certified-continuous-cover":
            from .continuous_cover import ContinuousCoverConfig,run_continuous_covers
            if args.config:
                if args.quick or args.seed is not None:
                    parser.error("--config cannot combine with --quick or --seed")
                config=ContinuousCoverConfig(**read_json(args.config))
            else:
                config=ContinuousCoverConfig.quick() if args.quick else ContinuousCoverConfig()
                if args.seed is not None:config.seed=args.seed
            print(json.dumps(run_continuous_covers(args.output,config),ensure_ascii=False,indent=2))
        elif args.command=="verify-certified-continuous-cover":
            from .continuous_cover import verify_continuous_covers
            print(json.dumps(verify_continuous_covers(args.directory),ensure_ascii=False,indent=2))
        elif args.command=="certified-continuous-realization":
            from .continuous_closed_realization import ClosedRealizationConfig,run_closed_realizations
            if args.config:
                if args.quick or args.seed is not None:
                    parser.error("--config cannot combine with --quick or --seed")
                config=ClosedRealizationConfig(**read_json(args.config))
            else:
                config=ClosedRealizationConfig.quick() if args.quick else ClosedRealizationConfig()
                if args.seed is not None:config.seed=args.seed
            print(json.dumps(run_closed_realizations(args.output,config),ensure_ascii=False,indent=2))
        elif args.command=="verify-certified-continuous-realization":
            from .continuous_closed_realization import verify_closed_realizations
            print(json.dumps(verify_closed_realizations(args.directory),ensure_ascii=False,indent=2))
        elif args.command=="certified-generic-grid":
            from .continuous_generic_realization import (
                GridRealizationConfig,benchmark_coupled_system,run_generic_grid)
            from .continuous_separation import ContinuousReLUSystem
            system=(ContinuousReLUSystem.from_dict(read_json(args.system))
                    if args.system else benchmark_coupled_system())
            config=(GridRealizationConfig(**read_json(args.config))
                    if args.config else GridRealizationConfig())
            print(json.dumps(run_generic_grid(args.output,system,config),ensure_ascii=False,indent=2))
        elif args.command=="verify-certified-generic-grid":
            from .continuous_generic_realization import verify_generic_grid
            print(json.dumps(verify_generic_grid(args.directory),ensure_ascii=False,indent=2))
        elif args.command=="certified-transition-lower":
            from .transition_overlap_lower import (
                TransitionLowerConfig,benchmark_transition_lower_system,run_transition_lower)
            from .continuous_separation import ContinuousReLUSystem
            system=(ContinuousReLUSystem.from_dict(read_json(args.system))
                    if args.system else benchmark_transition_lower_system())
            config=(TransitionLowerConfig(**read_json(args.config))
                    if args.config else TransitionLowerConfig())
            print(json.dumps(run_transition_lower(args.output,system,config),ensure_ascii=False,indent=2))
        elif args.command=="verify-certified-transition-lower":
            from .transition_overlap_lower import verify_transition_lower_run
            print(json.dumps(verify_transition_lower_run(args.directory),ensure_ascii=False,indent=2))
        elif args.command=="certified-multiswitch-lower":
            from .multiswitch_lower import (
                MultiSwitchConfig,benchmark_multiswitch_system,run_multiswitch_lower)
            from .continuous_separation import ContinuousReLUSystem
            system=(ContinuousReLUSystem.from_dict(read_json(args.system))
                    if args.system else benchmark_multiswitch_system())
            config=(MultiSwitchConfig(**read_json(args.config))
                    if args.config else MultiSwitchConfig())
            print(json.dumps(run_multiswitch_lower(args.output,system,config),ensure_ascii=False,indent=2))
        elif args.command=="verify-certified-multiswitch-lower":
            from .multiswitch_lower import verify_multiswitch_run
            print(json.dumps(verify_multiswitch_run(args.directory),ensure_ascii=False,indent=2))
        elif args.command=="certified-shifted-realization":
            from .shifted_realization import (
                ShiftedRealizationConfig,run_shifted_realizations)
            if args.config:
                if args.quick or args.seed is not None:
                    parser.error("--config cannot combine with --quick or --seed")
                config=ShiftedRealizationConfig(**read_json(args.config))
            else:
                config=(ShiftedRealizationConfig.quick()
                        if args.quick else ShiftedRealizationConfig())
                if args.seed is not None:config.seed=args.seed
            print(json.dumps(run_shifted_realizations(args.output,config),ensure_ascii=False,indent=2))
        elif args.command=="verify-certified-shifted-realization":
            from .shifted_realization import verify_shifted_realizations
            print(json.dumps(verify_shifted_realizations(args.directory),ensure_ascii=False,indent=2))
        elif args.command in ("run","research","graphs"):
            if args.command=="run":config_class,runner=Config,run
            elif args.command=="research":
                from .research_suite import ResearchConfig,run_research
                config_class,runner=ResearchConfig,run_research
            else:
                from .graph_experiment import GraphConfig,run_graphs
                config_class,runner=GraphConfig,run_graphs
            if args.config:
                if args.quick or args.epochs is not None or args.seed is not None:
                    parser.error("--config cannot combine with --quick, --epochs or --seed")
                config=config_class(**read_json(args.config))
            else:
                config=config_class.quick() if args.quick else config_class()
                if args.seed is not None:config.seed=args.seed
                if args.epochs is not None:config.epochs=args.epochs
            summary=runner(args.output,config)
            print(json.dumps({"output":str(args.output),"runtime_seconds":summary["runtime_seconds"]},ensure_ascii=False,indent=2))
        elif args.command in ("joint", "numeric", "raw-numeric", "dependence-numeric", "regression-numeric", "oblique-numeric", "causal-guided", "conservative-guided", "ood-guided", "graph-decoder", "distilled-pc", "factorized-graph", "cost-sensitive-graph", "symmetric-skeleton", "graph-global", "quadratic-readout", "manifold-intervention", "collateral-intervention", "end-to-end-mechanism", "node-context-graph", "factorized-node-context", "antisymmetric-direction", "pair-consistent", "active-intervention-graph", "active-end-to-end", "mechanism-search", "guided", "relational"):
            if args.command=="active-end-to-end":
                from .active_end_to_end_experiment import ActiveEndToEndConfig,run_active_end_to_end
                config=ActiveEndToEndConfig.quick() if args.quick else ActiveEndToEndConfig()
                if args.seed is not None:config.seed=args.seed
                result=run_active_end_to_end(args.output,args.source,config)
            elif args.command=="active-intervention-graph":
                from .active_intervention_experiment import ActiveInterventionConfig,run_active_intervention
                config=ActiveInterventionConfig.quick() if args.quick else ActiveInterventionConfig()
                if args.seed is not None:config.seed=args.seed
                result=run_active_intervention(args.output,config)
            elif args.command=="pair-consistent":
                from .pair_consistent_experiment import PairConsistentConfig,run_pair_consistent
                config=PairConsistentConfig.quick() if args.quick else PairConsistentConfig()
                if args.seed is not None:config.seed=args.seed
                result=run_pair_consistent(args.output,config)
            elif args.command=="antisymmetric-direction":
                from .antisymmetric_direction_experiment import AntisymmetricDirectionConfig,run_antisymmetric_direction
                config=AntisymmetricDirectionConfig.quick() if args.quick else AntisymmetricDirectionConfig()
                if args.seed is not None:config.seed=args.seed
                result=run_antisymmetric_direction(args.output,config)
            elif args.command=="factorized-node-context":
                from .factorized_node_context_experiment import FactorizedNodeContextConfig,run_factorized_node_context
                config=FactorizedNodeContextConfig.quick() if args.quick else FactorizedNodeContextConfig()
                if args.seed is not None:config.seed=args.seed
                result=run_factorized_node_context(args.output,config)
            elif args.command=="node-context-graph":
                from .node_context_experiment import NodeContextConfig,run_node_context
                config=NodeContextConfig.quick() if args.quick else NodeContextConfig()
                if args.seed is not None:config.seed=args.seed
                result=run_node_context(args.output,config)
            elif args.command=="end-to-end-mechanism":
                from .end_to_end_mechanism_experiment import EndToEndMechanismConfig,run_end_to_end_mechanism
                config=EndToEndMechanismConfig.quick() if args.quick else EndToEndMechanismConfig()
                if args.seed is not None:config.seed=args.seed
                result=run_end_to_end_mechanism(args.output,args.source,config)
            elif args.command=="collateral-intervention":
                from .collateral_intervention_experiment import CollateralInterventionConfig,run_collateral_intervention
                config=CollateralInterventionConfig.quick() if args.quick else CollateralInterventionConfig()
                if args.seed is not None:config.seed=args.seed
                result=run_collateral_intervention(args.output,args.source,config)
            elif args.command=="manifold-intervention":
                from .manifold_intervention_experiment import ManifoldInterventionConfig,run_manifold_intervention
                config=ManifoldInterventionConfig.quick() if args.quick else ManifoldInterventionConfig()
                if args.seed is not None:config.seed=args.seed
                result=run_manifold_intervention(args.output,args.source,config)
            elif args.command=="quadratic-readout":
                from .quadratic_intervention_experiment import QuadraticInterventionConfig,run_quadratic_intervention
                config=QuadraticInterventionConfig.quick() if args.quick else QuadraticInterventionConfig()
                if args.seed is not None:config.seed=args.seed
                result=run_quadratic_intervention(args.output,args.source,config)
            elif args.command=="graph-global":
                from .graph_global_experiment import GraphGlobalConfig,run_graph_global
                config=GraphGlobalConfig.quick() if args.quick else GraphGlobalConfig()
                if args.seed is not None:config.seed=args.seed
                result=run_graph_global(args.output,args.source,config)
            elif args.command=="symmetric-skeleton":
                from .symmetric_skeleton_experiment import SymmetricSkeletonConfig,run_symmetric_skeleton
                config=SymmetricSkeletonConfig.quick() if args.quick else SymmetricSkeletonConfig()
                if args.seed is not None:config.seed=args.seed
                result=run_symmetric_skeleton(args.output,args.source,config)
            elif args.command=="cost-sensitive-graph":
                from .cost_sensitive_graph_experiment import CostSensitiveGraphConfig,run_cost_sensitive_graph
                config=CostSensitiveGraphConfig.quick() if args.quick else CostSensitiveGraphConfig()
                if args.seed is not None:config.seed=args.seed
                result=run_cost_sensitive_graph(args.output,args.source,config)
            elif args.command=="factorized-graph":
                from .factorized_graph_experiment import FactorizedGraphConfig,run_factorized_graph
                config=FactorizedGraphConfig.quick() if args.quick else FactorizedGraphConfig()
                if args.seed is not None:config.seed=args.seed
                result=run_factorized_graph(args.output,args.source,config)
            elif args.command=="distilled-pc":
                from .distilled_pc_experiment import DistilledPCConfig,run_distilled_pc
                config=DistilledPCConfig.quick() if args.quick else DistilledPCConfig()
                if args.seed is not None:config.seed=args.seed
                result=run_distilled_pc(args.output,args.source,config)
            elif args.command=="graph-decoder":
                from .graph_decoder_experiment import GraphDecoderConfig,run_graph_decoder
                config=GraphDecoderConfig.quick() if args.quick else GraphDecoderConfig()
                if args.seed is not None:config.seed=args.seed
                result=run_graph_decoder(args.output,args.source,config)
            elif args.command=="mechanism-search":
                from .mechanism_search_experiment import MechanismSearchConfig,run_mechanism_search
                config=MechanismSearchConfig.quick() if args.quick else MechanismSearchConfig()
                if args.seed is not None:config.seed=args.seed
                result=run_mechanism_search(args.output,config)
            elif args.command=="numeric":
                from .numeric_experiment import run_numeric
                result=run_numeric(args.output,args.source,seed=291 if args.seed is None else args.seed,quick=args.quick)
            elif args.command in ("raw-numeric","dependence-numeric","regression-numeric","oblique-numeric"):
                from .raw_numeric_experiment import RawNumericConfig,run_raw_numeric
                config=RawNumericConfig.quick() if args.quick else RawNumericConfig()
                config.target_scope="dependence" if args.command=="dependence-numeric" else "regression" if args.command=="regression-numeric" else "all" if args.command=="oblique-numeric" else "base"
                if args.command=="oblique-numeric":
                    config.mapping_kind="oblique";config.seed=1092 if args.quick else 1193
                    if not args.quick:config.test_worlds=8192;config.test_pairs=2048
                if args.seed is not None:config.seed=args.seed
                result=run_raw_numeric(args.output,args.source,config)
            elif args.command in ("causal-guided","conservative-guided","ood-guided"):
                from .causal_guided_experiment import CausalGuidedConfig,run_causal_guided
                config=CausalGuidedConfig.quick() if args.quick else CausalGuidedConfig()
                if args.command=="conservative-guided":
                    config.selection_policy="conservative";config.seed=1492 if args.quick else 1493
                elif args.command=="ood-guided":
                    config.selection_policy="ood_conservative";config.guard_worlds=64 if args.quick else 512;config.seed=1592 if args.quick else 1593
                if args.seed is not None:config.seed=args.seed
                result=run_causal_guided(args.output,args.source,config)
            else:
                if args.command=="joint":
                    from .joint_experiment import JointConfig,run_joint
                    config_class,runner=JointConfig,run_joint
                elif args.command=="guided":
                    from .guided_experiment import GuidedExperimentConfig,run_guided
                    config_class,runner=GuidedExperimentConfig,run_guided
                else:
                    from .relational_experiment import RelationalConfig,run_relational
                    config_class,runner=RelationalConfig,run_relational
                config=config_class.quick() if args.quick else config_class()
                if args.seed is not None:config.seed=args.seed
                result=runner(args.output,args.source,config) if args.command=="guided" else runner(args.output,config)
            print(json.dumps(result,ensure_ascii=False,indent=2))
        elif args.command=="run-all":
            from .full_project import run_all
            print(json.dumps(run_all(args.output,args.quick,91 if args.seed is None else args.seed),ensure_ascii=False,indent=2))
        elif args.command=="inspect":
            print(json.dumps(read_json(args.directory/"summary.json"),ensure_ascii=False,indent=2))
        else:
            if args.command=="verify":verifier=verify
            elif args.command=="verify-research":
                from .research_verify import verify_research
                verifier=verify_research
            elif args.command=="verify-graphs":
                from .graph_experiment import verify_graphs
                verifier=verify_graphs
            elif args.command=="verify-joint":
                from .joint_experiment import verify_joint
                verifier=verify_joint
            elif args.command=="verify-numeric":
                from .numeric_experiment import verify_numeric
                verifier=verify_numeric
            elif args.command in ("verify-raw-numeric","verify-dependence-numeric","verify-regression-numeric","verify-oblique-numeric"):
                from .raw_numeric_experiment import verify_raw_numeric
                verifier=verify_raw_numeric
            elif args.command in ("verify-causal-guided","verify-conservative-guided","verify-ood-guided"):
                from .causal_guided_experiment import verify_causal_guided
                verifier=verify_causal_guided
            elif args.command=="verify-active-end-to-end":
                from .active_end_to_end_experiment import verify_active_end_to_end
                verifier=verify_active_end_to_end
            elif args.command=="verify-active-intervention-graph":
                from .active_intervention_experiment import verify_active_intervention
                verifier=verify_active_intervention
            elif args.command=="verify-pair-consistent":
                from .pair_consistent_experiment import verify_pair_consistent
                verifier=verify_pair_consistent
            elif args.command=="verify-antisymmetric-direction":
                from .antisymmetric_direction_experiment import verify_antisymmetric_direction
                verifier=verify_antisymmetric_direction
            elif args.command=="verify-factorized-node-context":
                from .factorized_node_context_experiment import verify_factorized_node_context
                verifier=verify_factorized_node_context
            elif args.command=="verify-node-context-graph":
                from .node_context_experiment import verify_node_context
                verifier=verify_node_context
            elif args.command=="verify-end-to-end-mechanism":
                from .end_to_end_mechanism_experiment import verify_end_to_end_mechanism
                verifier=verify_end_to_end_mechanism
            elif args.command=="verify-collateral-intervention":
                from .collateral_intervention_experiment import verify_collateral_intervention
                verifier=verify_collateral_intervention
            elif args.command=="verify-manifold-intervention":
                from .manifold_intervention_experiment import verify_manifold_intervention
                verifier=verify_manifold_intervention
            elif args.command=="verify-quadratic-readout":
                from .quadratic_intervention_experiment import verify_quadratic_intervention
                verifier=verify_quadratic_intervention
            elif args.command=="verify-graph-global":
                from .graph_global_experiment import verify_graph_global
                verifier=verify_graph_global
            elif args.command=="verify-symmetric-skeleton":
                from .symmetric_skeleton_experiment import verify_symmetric_skeleton
                verifier=verify_symmetric_skeleton
            elif args.command=="verify-cost-sensitive-graph":
                from .cost_sensitive_graph_experiment import verify_cost_sensitive_graph
                verifier=verify_cost_sensitive_graph
            elif args.command=="verify-factorized-graph":
                from .factorized_graph_experiment import verify_factorized_graph
                verifier=verify_factorized_graph
            elif args.command=="verify-distilled-pc":
                from .distilled_pc_experiment import verify_distilled_pc
                verifier=verify_distilled_pc
            elif args.command=="verify-graph-decoder":
                from .graph_decoder_experiment import verify_graph_decoder
                verifier=verify_graph_decoder
            elif args.command=="verify-mechanism-search":
                from .mechanism_search_experiment import verify_mechanism_search
                verifier=verify_mechanism_search
            elif args.command=="verify-guided":
                from .guided_experiment import verify_guided_experiment
                verifier=verify_guided_experiment
            elif args.command=="verify-relational":
                from .relational_experiment import verify_relational
                verifier=verify_relational
            else:
                from .full_project import verify_all
                verifier=verify_all
            print(json.dumps(verifier(args.directory),ensure_ascii=False,indent=2))
    except (ValueError,FileExistsError,FileNotFoundError,KeyError,TypeError) as exc:
        parser.exit(2,f"error: {exc}\n")

if __name__=="__main__":raise SystemExit(main())








