import os


class API_Inference:

    def __init__(self, args):
        pass

    @staticmethod
    def run(self, args):
        cmd = 'curl http://localhost:{}/generate -d \'{"prompt": {}, "temperature": {}}\''
        if args.enable_lora:
            cmd = 'curl http://localhost:{}/generate -d \'{"model: "sql-lora", "prompt": {}, "temperature": {}}\''
        
        os.system(cmd)
