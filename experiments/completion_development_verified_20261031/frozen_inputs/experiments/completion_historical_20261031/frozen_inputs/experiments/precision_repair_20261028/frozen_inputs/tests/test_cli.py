import json
import subprocess
import sys
import numpy as np


def test_installed_csv_entrypoints(tmp_path):
    x = np.linspace(-1, 1, 30)
    data = np.column_stack((x, 2*x+1, -3*x+2))
    csv = tmp_path/'train.csv'
    np.savetxt(csv, data, delimiter=',', header='x,a,b', comments='')
    cfg = tmp_path/'config.json'
    cfg.write_text(json.dumps(dict(population_size=4, epochs=1, use_rnn=False, use_gp=False,
        max_evaluations=80, structured_seeds=False, min_candidates=1, polish_candidates=0)))
    model, predictions = tmp_path/'model.json', tmp_path/'predictions.csv'
    subprocess.run([sys.executable,'-m','dimensr_mimo','fit','--data',str(csv),'--features','x',
                    '--targets','a,b','--config',str(cfg),'--output',str(model)],check=True,capture_output=True)
    subprocess.run([sys.executable,'-m','dimensr_mimo','predict','--data',str(csv),'--features','x',
                    '--model',str(model),'--output',str(predictions)],check=True,capture_output=True)
    result=np.loadtxt(predictions,delimiter=',',skiprows=1)
    np.testing.assert_allclose(result,data[:,1:],atol=1e-9)
