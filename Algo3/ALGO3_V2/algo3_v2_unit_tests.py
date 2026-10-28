"""Focused offline tests for COMP-2.0 experimental tail decision gates.
No network or RIT instance required.
"""
from __future__ import annotations
import importlib.util,unittest,argparse,time
from unittest.mock import patch
from pathlib import Path

spec=importlib.util.spec_from_file_location('compv2',Path(__file__).with_name('algo3_competition_engine_v2.py'))
mod=importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name]=mod
spec.loader.exec_module(mod)

class MockJournal:
 def __init__(self):self.events=[]
 def event(self,*args,**kwargs):self.events.append((args,kwargs))
class MockMarket:
 def __init__(self,books):self.books=books
 def latest_pair(self,age):return self.books,{'THOR_M':0.,'THOR_A':0.}
class MockClient:pass

class TestTail(unittest.TestCase):
 def setUp(self):
  self.tender={'tender_id':29,'action':'BUY','quantity':76000,'price':9.20}
  self.baseline=mod.PositionSnapshot(ts_ns=mod.mono_ns(),tick=100,m=0,a=0,realized_m=0,realized_a=0,unrealized_m=0,unrealized_a=0,nlv=0,fines=0,limits=None,securities=[])
  self.cfg=mod.RuntimeConfig(fees={'THOR_M':.01,'THOR_A':.005},rebates={},max_order=25000,api_rate=100,exec_delay_ms=0,gross_limit=200000,net_limit=100000,securities={})
  self.args=argparse.Namespace(tail_hold_seconds=5.,tail_round_end_buffer=5.,max_book_age=.15,tail_take_dollars=.025,tail_stop_dollars=.04,tail_monitor_interval=.02)
  self.j=MockJournal()
  self.snap=mod.PositionSnapshot(ts_ns=mod.mono_ns(),tick=100,m=19000,a=19000,realized_m=0,realized_a=0,unrealized_m=0,unrealized_a=0,nlv=0,fines=0,limits=None,securities=[])
 def invoke(self,bid):
  books={
   'THOR_M':mod._mock_book('THOR_M',[(bid-.01,12000)],[(bid+.02,10000)]),
   'THOR_A':mod._mock_book('THOR_A',[(bid,15000)],[(bid+.01,10000)]),
  }
  with patch.object(mod,'get_case',return_value={'status':'ACTIVE','tick':100,'ticks_per_period':300}),\
   patch.object(mod,'sample_position_fast',return_value=self.snap),\
   patch.object(mod,'current_fines',return_value=0.):
   return mod.monitor_experimental_tail(MockClient(),self.j,MockMarket(books),self.cfg,self.tender,self.baseline,76000,19000,.08,self.args,1)
 def test_take_profit(self):
  r=self.invoke(9.35)
  self.assertEqual(r[0],'take_profit_edge')
  self.assertGreater(r[2]-.08,.025)
 def test_stop_loss(self):
  r=self.invoke(9.18)
  self.assertEqual(r[0],'stop_loss_edge')
  self.assertLess(r[2]-.08,-.04)
 def test_unexpected_position(self):
  self.snap.a=18000
  r=self.invoke(9.35)
  self.assertEqual(r[0],'unexpected_inventory_change')
 def test_eligibility(self):
  a=argparse.Namespace(live=True,experimental_live_tail=True,acknowledge_uncalibrated_tail=True,tail_low_edge=.05,tail_high_edge=.10,tail_fraction=.25,tail_max_shares=25000)
  self.assertEqual(mod.experimental_tail_qty(.05,76000,a),19000)
  self.assertEqual(mod.experimental_tail_qty(.10,76000,a),0)
  a.experimental_live_tail=False
  self.assertEqual(mod.experimental_tail_qty(.075,76000,a),0)

if __name__=='__main__':unittest.main(verbosity=2)
