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
    def __init__(self, seed, P=92387, **kwargs):
        super(UHasher, self).__init__(seed)
        self.P = P
        self.gen = torch.Generator()
        self.gen.manual_seed(self.seed)
        print("Hashing using seed", seed)

        self.random_numbers = torch.randint(low=1, high=int(self.P/2) - 1, size=(4,), generator=self.gen)
        self.random_numbers = 2*self.random_numbers + 1
        

    def hash1(self, numbers, target_size=None):
        assert(target_size < self.P)
        self.random_numbers = self.random_numbers.to(numbers.device)
        return ((numbers * self.random_numbers[0] + torch.square(numbers) * self.random_numbers[1] + self.random_numbers[2]) % self.P) % target_size
        #return ((numbers * self.random_numbers[0] +  self.random_numbers[1]) % self.P) % target_size

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


class Mapper:
    def __init__(self, **kwargs):
        pass

    def get_general_idx(self, w_shape, original_offset, target_size, **kwargs):
      total_num = np.prod(w_shape)
      global_locations = torch.arange(total_num) + original_offset
      idx = (self.hasher.hash1(global_locations, target_size)) % target_size
      idx = idx.reshape(*w_shape)
      return idx

    def get_mlp_idx(self, **kwargs):
        return self.get_general_idx(**kwargs)

    def get_embedding_idx(self, **kwargs):
        return self.get_general_idx(**kwargs)
           
    def get_conv2d_idx(self, **kwargs):
        return self.get_general_idx(**kwargs)

    def get_general_g(self, w_shape, original_offset, target_size, **kwargs):
      total_num = np.prod(w_shape)
      global_locations = torch.arange(total_num) + original_offset
      g = 2*(self.hasher.hash1(global_locations, 2)) - 1
      g = g.reshape(*w_shape)
      return g

    
    def get_mlp_g(self, **kwargs):
        return self.get_general_g(**kwargs)

    def get_embedding_g(self, **kwargs):
        return self.get_general_g(**kwargs)

    def get_idx(self, mode, **kwargs):
        if mode == "mlp":
            return self.get_mlp_idx(**kwargs)
        if mode == "embedding":
            return self.get_embedding_idx(**kwargs)
        if mode == "conv2d":
            return self.get_conv2d_idx(**kwargs)

        return self.get_general_idx(**kwargs)

    def get_g(self, mode, **kwargs):
        if mode == "mlp":
            return self.get_mlp_g(**kwargs)
        if mode == "embedding":
            return self.get_embedding_g(**kwargs)

        return self.get_general_g(**kwargs)



class SparseRoastMapper(Mapper):
    def __init__(self, hasher, **kwargs):
      super(SparseRoastMapper, self).__init__()
      self.hasher = HasherFactory.get(hasher, **kwargs)
    
    def get_mlp_idx(self, w_shape, block_k, block_n, redn_factor, vectorization, **kwargs):
        assert(len(w_shape) == 2)
        w_shape = list(w_shape)
        w_shape[0], w_shape[1] = w_shape[1], w_shape[0]
    
        random_numbers = self.hasher.random_numbers
        assert(len(random_numbers) == 4)

        offset = self.get_sparse_idx(redn_factor, w_shape[0], w_shape[1], random_numbers[3], random_numbers[2], 
                                random_numbers[1], random_numbers[0], block_n, block_k, vectorization)

        idx = offset
        return np.transpose(idx)

    def get_sparse_idx(self, redn_factor,
                K : int, N: int,
                R3: int, R2: int, R1: int, R0: int,
                BLOCK_SIZE_N: int, BLOCK_SIZE_K: int, VEC: int,
           ):
          stride_bn = 1
          stride_bk = N
          idx = torch.zeros((K, N)).long()
          for pid_n in range((N + BLOCK_SIZE_N - 1)//BLOCK_SIZE_N):
              for k in range(0, (K + BLOCK_SIZE_K*redn_factor - 1) // (BLOCK_SIZE_K* redn_factor)):
                  for ck in range(0, redn_factor):
                      
                      block = (k* BLOCK_SIZE_K + (torch.arange(BLOCK_SIZE_K).long() + (BLOCK_SIZE_K - (((R2*pid_n + R1*(k+1) + R0*(ck+1) + R3) * VEC) % BLOCK_SIZE_K))) % BLOCK_SIZE_K).reshape(-1,1) * stride_bk + (pid_n * BLOCK_SIZE_N + torch.arange(BLOCK_SIZE_N).reshape(1,-1)) * stride_bn
                      off = k*BLOCK_SIZE_K* redn_factor + ck*BLOCK_SIZE_K
                      idx[off:off+BLOCK_SIZE_K,pid_n*BLOCK_SIZE_N:(pid_n+1)*BLOCK_SIZE_N] = block

          return idx



class MapperFactory:
    def get(mapper, hasher, seed, **kwargs):
        if mapper == "roast_sparse":
            return SparseRoastMapper(hasher=hasher, seed=seed, **kwargs)
        raise NotImplementedError
