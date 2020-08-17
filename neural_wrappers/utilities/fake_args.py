class FakeArgs:
    def __init__(self, Dict):
        for key in Dict:
            setattr(self, key, Dict[key])
