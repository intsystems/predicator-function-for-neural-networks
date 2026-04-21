
class MyClass:
    def __init__(self):
        pass
    def foo(self, a, b):
        print("Foo")

    def foo(self, a, b, c):
        print("Bar")

my_class = MyClass()
my_class.foo(1, 2)
