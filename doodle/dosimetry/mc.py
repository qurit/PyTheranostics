
import os

class MonteCarlo:
    def __init__(self, n_cpu, n_primaries, output_dir):
        self.n_cpu = n_cpu 
        self.n_primaries = n_primaries
        self.output_dir = output_dir

    def split_simulations(self):
        n_primaries_per_mac = int(self.n_primaries / self.n_cpu)
  

        with open("./main_template.mac",'r') as mac_file:
            filedata = mac_file.read()
        
        for i in range(0, self.n_cpu):
            new_mac = filedata
            
            new_mac = new_mac.replace('distrib-SPLIT.mhd',f'distrib_SPLIT_{i+1}.mhd')
            new_mac = new_mac.replace('stat-SPLIT.txt',f'stat__SPLIT_{i+1}.txt')    
            new_mac = new_mac.replace('XXX',str(n_primaries_per_mac))
            
            with open(f'{self.output_dir}/main_normalized_{i+1}.mac','w') as output_mac:
                output_mac.write(new_mac)
            
    def run_MC(self):
        os.system(f"bash {self.output_dir}/runsimulation1.sh {self.output_dir} {self.n_cpu}")
        