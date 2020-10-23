class AbstractMethodError(NotImplementedError):
    """
    Raise this error instead of NotImplementedError for abstract methods
    while keeping compatibility with Python 2 and Python 3.
    """

    def __init__(self, class_instance, methodtype="method"):
        types = {"method", "classmethod", "staticmethod", "property"}
        if methodtype not in types:
            raise ValueError(
                f"methodtype must be one of {types}, got {methodtype} instead."
            )
        self.methodtype = methodtype
        self.class_instance = class_instance

    def __str__(self) -> str:
        if self.methodtype == "classmethod":
            name = self.class_instance.__name__
        else:
            name = type(self.class_instance).__name__
        return f"This {self.methodtype} must be defined in the concrete class {name}"
