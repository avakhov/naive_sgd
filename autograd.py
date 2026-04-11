class Func:
    @classmethod
    def create_const(cls, a):
        return cls("const", a)

    def __init__(self, kind, value=0.0):
        self.kind = kind
        self.value = value

print(Func.create_const(123))
