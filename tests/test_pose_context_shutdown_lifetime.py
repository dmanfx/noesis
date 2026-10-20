"""The World actor is not the owner of live SGIE restoration probes."""
import ast
import logging
import json
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
from noesis.pipelines import hooks


def runtime_finalizer():
    # Execute the ACTUAL SDK-neutral runtime helper without importing the host
    # entrypoint (which performs native attestation/module loading at import).
    source=Path(hooks.__file__).parents[1]/'ds9_runtime_core.py'
    tree=ast.parse(source.read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_finalize_pose_roi_context')
    namespace={'json':json}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),namespace)
    return namespace['_finalize_pose_roi_context'],tree


def finalize(pipeline, **changes):
    function,_=runtime_finalizer();state={}
    args=dict(eos_accepted=True,eos_seen=True,wait_completed=True,wait_failed=False)
    args.update(changes)
    result=function(pipeline,**args,runtime_state=state,logger=logging.getLogger('test.pose.shutdown'))
    return result,state


class Controller:
    def __init__(self): self.closed=False;self.restored=0;self.prepared=1
    def restore_inflight(self):
        assert not self.closed, 'live SGIE restoration probe was removed'
        self.restored+=1
    def close(self): self.closed=True
    def snapshot(self):return dict(copied=1,expanded=self.prepared,restored=self.restored,failures=0,elapsed_ns=0)


def test_r91_real_world_worker_close_leaves_live_context_for_inflight_restore():
    controller=Controller();events=[]
    pipeline=NS(pose_roi_context_controller=controller)
    processor=NS(pipeline=pipeline)
    worker=hooks._TrackingPublicationWorker(processor,world_close=lambda:events.append('world_closed'))
    worker.shutdown(timeout_s=2.)
    assert events==['world_closed'] and worker.close_receipt.completed
    assert not controller.closed
    controller.restore_inflight()
    result,state=finalize(pipeline)
    assert result and controller.closed
    assert state['pose_roi_context_shutdown_receipt']=={'present':True,'quiesced':True}


@pytest.mark.parametrize('changes',[
    {'eos_accepted':False},{'eos_seen':False},{'wait_completed':False},{'wait_failed':True},
    {'wait_completed':1},
])
def test_r91_unproven_native_pipeline_retains_paired_probes(changes):
    c=Controller();p=NS(pose_roi_context_controller=c)
    result,state=finalize(p,**changes)
    assert not result and state['pipeline_failed'] and not c.closed
    assert state['pose_roi_context_shutdown_receipt']=={'present':True,'quiesced':False}
    assert p.pose_roi_context_controller is c
    c.restore_inflight()


@pytest.mark.parametrize('stage',['close','snapshot'])
def test_r91_native_close_or_summary_failure_is_not_suppressed(stage):
    c=Controller();p=NS(pose_roi_context_controller=c)
    def fail():raise RuntimeError('owned context '+stage+' failure')
    setattr(c,stage,fail)
    result,state=finalize(p)
    assert not result and state['pipeline_failed']
    assert state['pose_roi_context_shutdown_receipt']['quiesced'] is False
    assert p.pose_roi_context_controller is c


def test_r91_absent_default_off_context_needs_no_native_resource():
    result,state=finalize(NS())
    assert result and 'pipeline_failed' not in state
    assert state['pose_roi_context_shutdown_receipt']=={'present':False,'quiesced':True}


def test_r91_actual_runtime_call_binds_all_native_quiescence_facts():
    function,tree=runtime_finalizer()
    owner=next(n for n in ast.walk(tree) if isinstance(n,ast.If)
        and ast.unparse(n.test)=='not pipeline_quiesced'
        and any(isinstance(x,ast.Call) and isinstance(x.func,ast.Name)
                and x.func.id=='_finalize_pose_roi_context' for x in ast.walk(ast.Module(body=n.orelse,type_ignores=[]))))
    call=next(n for n in ast.walk(ast.Module(body=owner.orelse,type_ignores=[]))
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='_finalize_pose_roi_context')
    c=Controller();state={};p=NS(pose_roi_context_controller=c)
    env=dict(_finalize_pose_roi_context=function,pipeline=p,eos_accepted=True,eos_seen=True,
        wait_completed=False,wait_failed=False,runtime_state=state,logger=logging.getLogger('test.runtime'))
    # Exercise the actual call arguments, not a hand-written facsimile. Even
    # an erroneously invoked finalizer may not substitute success for wait.
    assert eval(compile(ast.Expression(call),'<actual-runtime-call>','eval'),env) is False
    assert not c.closed
    c.restore_inflight();env['wait_completed']=True
    assert eval(compile(ast.Expression(call),'<actual-runtime-call>','eval'),env) is True
    assert c.closed and c.restored==c.prepared
