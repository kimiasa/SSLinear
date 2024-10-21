import torch
import numpy as np
import mmh3 

''' hasher classes that can be used in actual kernels '''
class Hasher:
    def __init__(self, seed):
      self.seed = seed

    def hash(self, numbers):
      raise NotImplementedError


class MurmurHasher(Hasher):
    def __init__(self, seed, **kwargs):
        super(MurmurHasher, self).__init__(seed)
        self.seed = seed

    def hash1(self, numbers, target_size=None):
        device = numbers.device
        cpu_numbers = np.array(numbers.to("cpu"))
        hashed_numbers = np.array([ mmh3.hash(i, seed=self.seed) for i in cpu_numbers])
        hashed_numbers = torch.LongTensor(hashed_numbers).to(device) % target_size
        return hashed_numbers

    def hash2(self, number1, number2, target_size=None):
        device = number1.device
        output = np.zeros(shape=(number1.numel(), number2.numel()))
        cpu_num1 = np.array(number1.to("cpu")).reshape(-1)
        cpu_num2 = np.array(number2.to("cpu")).reshape(-1)
        for num1 in range(len(cpu_num1)):
            for num2 in range(len(cpu_num2)):
                output[num1,num2] = mmh3.hash(str(cpu_num1[num1])+"-"+str(cpu_num2[num2]), seed=self.seed, signed=False)
        return torch.LongTensor(output).to(device) %target_size

class UHasher(Hasher):
    def __init__(self, seed, P=45007, **kwargs):
        super(UHasher, self).__init__(seed)
        self.P = P
        self.gen = torch.Generator(device=torch.get_default_device())
        self.gen.manual_seed(self.seed)

        self.random_numbers = torch.randint(low=1, high=int(self.P/2) - 1, size=(4,), generator=self.gen)
        self.random_numbers = 2*self.random_numbers + 1
        

    def hash1(self, numbers, target_size=None):
        assert(target_size < self.P)
        self.random_numbers = self.random_numbers.to(numbers.device)
        return ((numbers * self.random_numbers[0] + torch.square(numbers) * self.random_numbers[1] + self.random_numbers[2]) % self.P) % target_size

    def hash2(self, number1, number2, target_size=None):
        assert(target_size < self.P)
        self.random_numbers = self.random_numbers.to(number1.device)
        return ((number1 * self.random_numbers[0] + number2 * self.random_numbers[1] + self.random_numbers[2]) % self.P) % target_size

class HasherFactory:
    def get(hasher, seed, **kwargs):
        if hasher == "uhash":
            return UHasher(seed, **kwargs)
        if hasher == "mhash":
            return MurmurHasher(seed, **kwargs)
        raise NotImplementedError



