from abc import abstractmethod                                                   
from contracts import contract, with_metaclass, ContractsMeta                    

class DIISInterface(with_metaclass(ContractsMeta, object)):

  @abstractmethod
  @contract(array_error_pairs = 'seq[>0](tuple[2])')
  def add_entry(self, *array_error_pairs):
    pass

  @abstractmethod
  @contract(returns = 'seq[>0]')
  def extrapolate(self):
    pass


if __name__ == "__main__":
  a = DIISInterface()
