"""Safety checks for planning, bounded loading, downloads, and narrowly scoped cleanup."""
import hashlib
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import air_bench as b


class Response(io.BytesIO):
    def __init__(self, data=b'', size=0):
        super().__init__(data)
        self.headers = {'Content-Length': str(size)}


class AirBenchTests(unittest.TestCase):
    def test_report_explains_slower_mlx_without_disqualifying_comparison(self):
        report = dict(rows=5_000_000,iterations=15,mlx_fit_including_labels_s=2.575,
                      mlx_inertia_float64=100.,load_s=.1,prepare_s=.2,
                      sklearn=dict(fit_s=2.013,iterations=40,inertia_float64=101.))
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            b.print_results(report)
        text = output.getvalue()
        self.assertIn('MLX took 28% longer',text)
        self.assertIn('SAME data',text)
        self.assertIn('0.99% lower',text)
        self.assertNotIn('not an equal-work',text)
        self.assertIn('effectively tied',b.timing_verdict(1,1.005))

    def test_common_quality_metric_and_controlled_check(self):
        import numpy as np
        X=np.random.default_rng(5).normal(size=(100,6)).astype(np.float32)
        centers=X[:8].copy()
        labels=np.zeros(100,dtype=np.int64)
        expected=float(((X.astype(np.float64)-centers[0])**2).sum())
        self.assertAlmostEqual(b.inertia_for_labels(X,centers,labels),expected)
        with contextlib.redirect_stdout(io.StringIO()):
            result=b.controlled_comparison(X)
        for name in ['mlx','sklearn']:
            self.assertEqual(len(result['raw_seconds'][name]),3)
            self.assertGreater(result['median_seconds'][name],0)
        values=result['inertia_float64']
        self.assertLess(abs(values['mlx']/values['sklearn']-1),1e-4)

    def test_memory_plans(self):
        for ram in [8,16,24,128]:
            p = b.plan_rows(ram*b.GIB,ram*b.GIB//2,ram*b.GIB*.7)
            self.assertLessEqual(p['rows'],5_000_000)
            self.assertLessEqual(p['estimated_working_bytes'],p['memory_budget_bytes'])
            self.assertLessEqual(p['memory_budget_bytes'],ram*b.GIB*.25)
        p = b.plan_rows(16*b.GIB,10*b.GIB,10*b.GIB,memory_mib=300)
        self.assertLess(p['rows'],200_000)
        with self.assertRaises(ValueError):
            b.plan_rows(8*b.GIB,100*b.MIB,4*b.GIB)

    def test_cleanup_preserves_unrelated_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache=Path(tmp).resolve()/'cache'
            b.owned_cache(cache,True)
            (cache/b.NAME).write_bytes(b'owned download')
            (cache/'my-data.txt').write_text('keep')
            b.cleanup(cache)
            self.assertEqual((cache/'my-data.txt').read_text(),'keep')
            self.assertFalse((cache/b.NAME).exists())

    def test_cleanup_rejects_unowned_and_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve()
            cache=root/'cache'; cache.mkdir()
            (cache/b.NAME).write_bytes(b'user data')
            with self.assertRaises(ValueError): b.cleanup(cache)
            self.assertTrue((cache/b.NAME).exists())
            link=root/'link'; link.symlink_to(cache,target_is_directory=True)
            with self.assertRaises(ValueError): b.cleanup(link)

    def test_download_and_cache_integrity(self):
        data=b'small fixture'
        with tempfile.TemporaryDirectory() as tmp:
            cache=Path(tmp).resolve()/'cache'
            with patch.object(b.urllib.request,'urlopen',side_effect=[Response(size=len(data)),Response(data)]):
                dest,info=b.fetch(cache)
            self.assertEqual(info['sha256'],hashlib.sha256(data).hexdigest())
            with patch.object(b.urllib.request,'urlopen',side_effect=AssertionError('no network expected')):
                self.assertEqual(b.fetch(cache)[0],dest)
            dest.write_bytes(b'corrupt')
            with self.assertRaises(ValueError): b.fetch(cache)

    def test_oversized_and_incomplete_download(self):
        for size,data in [(b.MAX_DOWNLOAD+1,b''),(10,b'short'),(2,b'too long')]:
            with tempfile.TemporaryDirectory() as tmp:
                cache=Path(tmp).resolve()/'cache'
                with patch.object(b.urllib.request,'urlopen',side_effect=[Response(size=size),Response(data)]):
                    with self.assertRaises(ValueError): b.fetch(cache)
                self.assertFalse((cache/b.NAME).exists())
                self.assertFalse((cache/(b.NAME+'.partial')).exists())

    def test_loader_stops_at_requested_rows(self):
        import numpy as np
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'data.parquet'
            n=100
            start=np.full(n,np.datetime64('2015-01-01T10:00:00'))
            table=pa.table(dict(tpep_pickup_datetime=start,tpep_dropoff_datetime=start+np.timedelta64(10,'m'),
                               trip_distance=np.ones(n),fare_amount=np.full(n,10.),tip_amount=np.full(n,2.),
                               passenger_count=np.ones(n)))
            pq.write_table(table,path,row_group_size=20)
            X,scanned=b.load_features(path,10)
            self.assertEqual(X.shape,(10,6))
            np.testing.assert_allclose(X[0],[1,10,10,20,10,1])


if __name__=='__main__':
    unittest.main()
