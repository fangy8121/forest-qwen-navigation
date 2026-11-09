"""通过 JSON 行协议调用独立模型环境，供异步规划线程使用。"""
import json
from pathlib import Path
import subprocess


class ModelProcess:
    def __init__(self, python, cfg, output):
        self.log=(output/'model_worker.log').open('w')
        executable=str(Path(python).expanduser().absolute())
        command=[executable, '-m', 'omni_drones.navigation.forest.model_worker',
            '--model',str(Path(cfg.model).resolve()),'--device',cfg.qwen_device,
            '--max-input-tokens',str(cfg.max_input_tokens),'--max-new-tokens',str(cfg.max_new_tokens)]
        if cfg.adapter:
            command += ['--adapter',str(Path(cfg.adapter).resolve())]
        self.process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,
            stderr=self.log,text=True,bufsize=1)
        self.metadata=self._receive()['metadata']
        self.metadata['worker_python']=executable

    def _receive(self):
        line=self.process.stdout.readline()
        if not line:
            raise RuntimeError('model worker exited; see model_worker.log')
        result=json.loads(line)
        if 'error' in result:
            raise RuntimeError(result['error'])
        return result

    def generate(self, image, prompt):
        self.process.stdin.write(json.dumps(dict(image=str(Path(image).resolve()),prompt=prompt))+'\n')
        self.process.stdin.flush()
        result=self._receive()
        return result['raw'],result['latency_seconds']

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            self.process.wait()
        self.log.close()
