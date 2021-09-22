from pathlib import Path
import json

this_dir=Path(__file__).resolve().parent.parent
ISOTOPE_DATA_FILE = Path(this_dir,"isotope_data","isotopes.json")


class QC:
    def __init__(self,isotope):

        with open(ISOTOPE_DATA_FILE) as f:
            self.isotope_dic = json.load(f)

        self.isotope = isotope
        self.isotope_dic = self.isotope_dic[isotope]
        self.summary = ''

    
    def append_to_summary(self,text):
        self.summary = self.summary + text
    
    def print_summary(self):
        print(self.summary)

             

        