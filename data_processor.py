import os
import pandas as pd
import xml.etree.ElementTree as ET
import zipfile
from glob import glob
from tqdm import tqdm

class MedQuADProcessor:
    """Processes MedQuAD dataset from XML files into a structured DataFrame"""
    
    def __init__(self, zip_path):
        self.zip_path = zip_path
        self.extraction_path = os.path.join(os.path.dirname(zip_path), 'MedQuAD-extracted')
        
    def extract_dataset(self):
        """Extracts the MedQuAD dataset from the zip file"""
        if not os.path.exists(self.extraction_path):
            print(f"Extracting dataset to {self.extraction_path}...")
            with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.extraction_path)
            print("Extraction complete!")
        else:
            print(f"Using previously extracted dataset at {self.extraction_path}")
    
    def parse_xml_file(self, xml_file):
        """Parses a single XML file containing Q&A pairs"""
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            qa_pairs = []

            for qa_pair in root.findall('.//QAPair'):
                question_text = qa_pair.findtext('Question/QuestionText', default="")
                answer_text = qa_pair.findtext('Answer/AnswerText', default="")

                if question_text and answer_text:
                    qa_pairs.append({
                        'question': question_text.strip(),
                        'answer': answer_text.strip()
                    })

            return qa_pairs
        except Exception as e:
            print(f"Error parsing {xml_file}: {e}")
            return []

    def process_all_xml_files(self):
        """Processes all XML files in the dataset"""
        self.extract_dataset()
        xml_files = glob(os.path.join(self.extraction_path, '**', '*.xml'), recursive=True)
        print(f"Found {len(xml_files)} XML files.")

        all_qa_pairs = []
        for xml_file in tqdm(xml_files, desc="Processing XML files"):
            all_qa_pairs.extend(self.parse_xml_file(xml_file))

        df = pd.DataFrame(all_qa_pairs)

        # Standardize column names
        df.columns = df.columns.str.lower()

        # Remove duplicates and empty answers
        df.drop_duplicates(subset=['question'], keep="first", inplace=True)
        df.dropna(subset=['answer'], inplace=True)

        return df

    def save_dataset(self, df, output_path='data/medquad_cleaned.csv'):
        """Saves the processed dataset"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Dataset saved to {output_path}")
        return output_path

if __name__ == "__main__":
    processor = MedQuADProcessor("C:/Users/swathiga/Downloads/MedQuAD-master (1).zip")
    df = processor.process_all_xml_files()
    processor.save_dataset(df)
