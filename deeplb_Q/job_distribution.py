import numpy as np


class Dist:

    def __init__(self, num_q, max_job_len):
        self.num_q = num_q
        self.max_job_len = max_job_len

        self.job_small_chance = 0.7

        self.job_len_big_lower = max_job_len * 2 / 3
        self.job_len_big_upper = max_job_len

        self.job_len_small_lower = 1
        self.job_len_small_upper = max_job_len / 5

    
    def light_tailed(self):
        """
        Inverse CDF given by P(s > k) = p ** k for p in (0, 1)
        """
        return

    def heavy_tailed(self):
        """
        Inverse CDF given by P(s > k) = c / (k ** alpha) for alpha in (1, 2)
        """
        return

    def normal_dist(self):
        """
        Jobs have a duration uniformly b/t [1, job_len]
        """
        # new work duration
        nw_len = np.random.randint(1, self.max_job_len + 1)

        return nw_len


    def bi_model_dist(self):
        """
        Jobs have a small duration with prob job_small_chance and a larger
        duration with prob 1 - job_small chance. 
        """

        if np.random.rand() < self.job_small_chance: # small job
            nw_len = np.random.randint(self.job_len_small_lower,
                                       self.job_len_small_upper + 1)
        else:  # big job
            nw_len = np.random.randint(self.job_len_big_lower,
                                       self.job_len_big_upper + 1)

        return nw_len


def generate_sequence_work(pa, seed=42):
    """
    Generate jobs for all job sequences according to nw_dist
    """

    np.random.seed(seed)
    simu_len = pa.simu_len * pa.num_ex
    nw_dist = pa.dist.bi_model_dist
    nw_len_seq = np.zeros(simu_len, dtype=int)

    for i in range(simu_len):
        if np.random.rand() < pa.new_job_rate:  # a new job comes
            nw_len_seq[i] = nw_dist()

    # reshape job lengths into matrix where rows are job sequences
    nw_len_seq = np.reshape(nw_len_seq,
                            [pa.num_ex, pa.simu_len])

    return nw_len_seq
