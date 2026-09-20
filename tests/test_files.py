import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from mlx_kmeans import KMeans, cluster_file
from mlx_kmeans.files import inspect_file


class FileTests(unittest.TestCase):
    def test_real_fit_preserves_ids_and_reports_original_means(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'results'
            result=cluster_file(ROOT/'examples/customers.csv',columns=['annual_spend','orders','visits'],
                                k=3,n_init=2,output=out)
            labeled=pd.read_csv(out/'labeled.csv',dtype={'customer_id':str})
            self.assertEqual(labeled.customer_id.iloc[0],'001')
            self.assertEqual(sorted(labeled.cluster.unique()),[1,2,3])
            actual=labeled.groupby('cluster').annual_spend.mean().to_numpy()
            np.testing.assert_allclose(actual,result['summary'].mean_annual_spend)
            self.assertEqual(result['summary'].rows.sum(),12)
            self.assertTrue((out/'model.npz').exists())
            with self.assertRaisesRegex(ValueError,'already exists'):
                cluster_file(ROOT/'examples/customers.csv',columns=['orders'],k=3,output=out)

    def test_missing_rows_require_explicit_choice_and_retain_alignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'input.csv'
            source.write_text('id,x,cluster\n001,0,old\n002,,old\n003,10,old\n004,11,old\n')
            with self.assertRaisesRegex(ValueError,'1 rows'):
                cluster_file(source,columns=['x'],k=2)
            result=cluster_file(source,columns=['x'],k=2,n_init=1,missing='drop')
            labeled=pd.read_csv(result['output']/'labeled.csv',dtype={'id':str})
            self.assertTrue(pd.isna(labeled.kmeans_cluster.iloc[1]))
            self.assertEqual(labeled.cluster.tolist(),['old']*4)
            self.assertEqual(result['run']['excluded_rows'],1)

    def test_excel_sheet_and_inspection(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'sample.xlsx'
            frame=pd.DataFrame({'id':['001','002','003','004'],'x':[0,1,9,10]})
            frame.to_excel(path,index=False,sheet_name='Customers')
            self.assertEqual(inspect_file(path,'Customers').id.iloc[0],'001')
            result=cluster_file(path,columns=['x'],k=2,n_init=1,sheet='Customers')
            self.assertEqual(result['run']['clustered_rows'],4)
            with self.assertRaisesRegex(ValueError,'Available: Customers'):
                inspect_file(path,'Missing')

    def test_duplicate_header_and_low_memory_rejected(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'bad.csv';p.write_text('x,x\n1,2\n')
            with self.assertRaisesRegex(ValueError,'unique'):
                inspect_file(p)
            with patch('psutil.virtual_memory',return_value=SimpleNamespace(total=2**30,available=1)):
                with self.assertRaisesRegex(ValueError,'budget'):
                    cluster_file(ROOT/'examples/customers.csv',columns=['orders'],k=2)

    def test_final_inertia_and_restart_selection(self):
        X=np.array([[0.,0.],[1.,0.],[9.,0.],[10.,0.]],np.float32)
        fitted=KMeans(n_clusters=2,max_iter=1,random_state=3).fit(X)
        expected=np.min(((X[:,None]-fitted.cluster_centers_)**2).sum(2),axis=1).sum()
        self.assertAlmostEqual(fitted.inertia_,float(expected),places=5)
        good=np.array([[.5,0],[9.5,0]],np.float32)
        bad=np.array([[0,0],[1,0]],np.float32)
        with patch.object(KMeans,'_one_run',side_effect=[(bad,0.,1),(good,999.,1)]):
            fitted=KMeans(n_clusters=2,n_init=2).fit(X)
        np.testing.assert_equal(fitted.cluster_centers_,good)
        self.assertAlmostEqual(fitted.inertia_,1.)
        for kwargs in [dict(max_iter=0),dict(n_init=0),dict(n_clusters=0)]:
            with self.assertRaises(ValueError): KMeans(**kwargs).fit(X)

    def test_cli_no_args_is_safe_and_preview_does_not_fit(self):
        r=subprocess.run([sys.executable,'-m','mlx_kmeans'],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(r.returncode,0)
        self.assertIn('usage:',r.stdout)
        r=subprocess.run([sys.executable,'-m','mlx_kmeans','examples/customers.csv'],cwd=ROOT,
                         capture_output=True,text=True)
        self.assertEqual(r.returncode,0)
        self.assertIn('customer_id',r.stdout)
        self.assertNotIn('Saved',r.stdout)


if __name__=='__main__':
    unittest.main()
