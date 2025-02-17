import os
import sys
import gc
import csv
import json
import copy
import logging
from datetime import datetime

import torch
import yaml
import pandas as pd
from tqdm import tqdm

from datasets import Dataset
from transformers import pipeline, BitsAndBytesConfig, AutoTokenizer
import transformers

import outlines
from outlines.processors.structured import JSONLogitsProcessor

from typing import Literal, Dict, List, get_origin, get_args
from pydantic import BaseModel


# -----------------------
# Pydantic Model Classes
# -----------------------
class StatusBase(BaseModel):
    ref: str
    status: str


# Metastasis related classes
class LymphNodeStatus(StatusBase):
    status: Literal["LN0", "LN1", "LN2"]


class HematogenousStatus(StatusBase):
    status: Literal["H0", "H1", "H2"]


class PeritonealStatus(StatusBase):
    status: Literal["P0", "P1", "P2"]


class MetastasisAssessment(BaseModel):
    lymph_node_metastasis: List[Dict[int, LymphNodeStatus]]
    final_lymph_node_status: Literal["LN0", "LN1", "LN2"]
    hematogenous_metastasis: List[Dict[int, HematogenousStatus]]
    final_hematogenous_metastasis_status: Literal["H0", "H1", "H2"]
    peritoneal_seeding: List[Dict[int, PeritonealStatus]]
    final_peritoneal_seeding_status: Literal["P0", "P1", "P2"]


# Vessel related classes
class ArteryStatus(StatusBase):
    status: Literal["A0", "A1", "A2", "A9"]


class VeinStatus(StatusBase):
    status: Literal["V0", "V1", "V2", "V9"]


class VesselStatus(StatusBase):
    status: Literal["A0", "A1", "A2", "A9", "V0", "V1", "V2", "V9"]


class IVCStatus(StatusBase):
    status: Literal["I0", "I1", "I2", "I9"]


class ArterialVariationStatus(StatusBase):
    status: Literal["VAR1", "VAR9"]


class VascularAssessment(BaseModel):
    branches_and_other: Dict[str, VesselStatus] = {}
    artery: Dict[str, ArteryStatus] = {
        "celiac_axis": ArteryStatus(ref="", status="A0"),
        "common_hepatic_artery": ArteryStatus(ref="", status="A0"),
        "superior_mesenteric_artery": ArteryStatus(ref="", status="A0")
    }
    aorta: ArteryStatus = ArteryStatus(ref="", status="A0")
    arterial_variations: ArterialVariationStatus = ArterialVariationStatus(ref="", status="VAR1")
    vein: Dict[str, VeinStatus] = {
        "main_portal_vein": VeinStatus(ref="", status="V0"),
        "superior_mesenteric_vein": VeinStatus(ref="", status="V0")
    }
    inferior_vena_cava: IVCStatus = IVCStatus(ref="", status="I0")


# Lymph Node Assessment
class LNAssessment(BaseModel):
    hematogenous_metastasis: str
    peritoneal_seeding: str
    tumor_site: str
    regional_lymph_node: LymphNodeStatus
    distant_lymph_node: LymphNodeStatus


# Tumor Assessment
class TumorMetastasisStatus(BaseModel):
    ref: str
    status: Literal["D0", "D1"]


class TumorAssessment(BaseModel):
    metastasis: str
    lymph_nodes: str
    vessel_involvement: str
    tumor_location: str
    tumor_size: str  # Only a single number is allowed
    tumor_morphology: str
    main_pancreatic_duct: TumorMetastasisStatus
    bile_duct: TumorMetastasisStatus
    adjacent_organ_invasion: str


# -----------------------
# Helper Functions
# -----------------------
def load_config(config_file):
    try:
        with open(config_file, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        logging.error(f"Error loading config file: {str(e)}")
        raise


def extract_answer(output):
    try:
        return json.loads(output)
    except json.JSONDecodeError as e:
        logging.error(f"JSON decoding failed: {str(e)}")
        return {}


def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    df['report'] = df['report'].astype(str)
    df.reset_index(inplace=True)  # This adds an 'index' column
    return Dataset.from_pandas(df[['index', 'report']])


def analyze_medical_report(example, instruct, pipe, logit):
    report = example['report']
    if "llama" in pipe.model.config.model_type.lower():
        prompt = [
            {"role": "system", "content": "You are an accurate and factual assistant that replies in JSON."},
            {"role": "user", "content": f"{instruct}\n\nReport:\n{report}\n\n[OUTPUT]\n"}
        ]
    else:
        prompt = [{"role": "user", "content": f"{instruct}\n\nReport:\n{report}\n\n[OUTPUT]\n"}]

    logit_loop = copy.deepcopy(logit)
    output = pipe(
        prompt,
        do_sample=False,
        max_new_tokens=500,
        truncation=True,
        return_full_text=True,
        logits_processor=transformers.LogitsProcessorList([logit_loop])
    )
    cot_output = output[0]['generated_text'][-1]['content']
    print(cot_output)
    answer = extract_answer(cot_output)
    example['prediction'] = answer
    example['chain_of_thought_output'] = cot_output
    example['input_prompt'] = prompt[0]['content']
    return example


def process_dataset(dataset, instruct_item, pipe, logit):
    instruct = instruct_item['prompt']
    results = []
    for example in tqdm(dataset, desc="Processing reports"):
        processed_example = analyze_medical_report(example, instruct, pipe, logit)
        results.append(processed_example)
    return results


def collect_model_fields(schema: BaseModel, parent_key: str = '') -> set:
    """
    Recursively collect all field names from a Pydantic model.
    """
    keys = set()
    for field_name, field in schema.__fields__.items():
        full_key = f"{parent_key}.{field_name}" if parent_key else field_name
        annotation = field.annotation
        origin = get_origin(annotation)
        args = get_args(annotation)

        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            keys.update(collect_model_fields(annotation, full_key))
        elif origin in (list, List):
            if args:
                first_arg = args[0]
                if isinstance(first_arg, type) and issubclass(first_arg, BaseModel):
                    keys.update(collect_model_fields(first_arg, full_key))
        elif origin in (dict, Dict):
            if len(args) == 2:
                value_type = args[1]
                if isinstance(value_type, type) and issubclass(value_type, BaseModel):
                    keys.update(collect_model_fields(value_type, full_key))
        keys.add(full_key)
    return keys


def load_existing_output(output_csv, output_keys):
    if os.path.exists(output_csv):
        df = pd.read_csv(output_csv)
    else:
        df = pd.DataFrame(columns=output_keys)
    return df


def update_output_csv(df, data_per_report, output_csv):
    new_data_df = pd.DataFrame.from_dict(data_per_report, orient='index')
    df = pd.concat([df, new_data_df]).drop_duplicates(subset=['index', 'model_name'], keep='last')
    df.to_csv(output_csv, index=False)


def flatten_dict(d, parent_key='', sep='.'):
    """
    Flatten a nested dictionary.
    """
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


# -----------------------
# Main Processing Block
# -----------------------
if __name__ == "__main__":
    try:
        config_file = "pdac_slm_singlescript.yaml"
        config = load_config(config_file)

        required_keys = ['models', 'input_csv', 'instruct']
        for key in required_keys:
            if key not in config:
                raise KeyError(f"Missing required key in config: {key}")

        if not os.path.exists(config['input_csv']):
            raise FileNotFoundError(f"Input CSV file not found: {config['input_csv']}")

        base_log_dir = os.path.join("analysis_logs", datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(base_log_dir, exist_ok=True)

        dataset = load_dataset(config['input_csv'])

        output_csv = os.path.join(base_log_dir, 'output.csv')
        full_log_csv = os.path.join(base_log_dir, 'full_log.csv')

        all_collect_columns = set()
        for instruct_item in config['instruct']:
            all_collect_columns.update(instruct_item.get('collect_columns', []))
        all_collect_columns = sorted(all_collect_columns)
        output_keys = ['index', 'model_name'] + all_collect_columns

        full_log_keys = set(['index', 'model_name', 'instruct_name', 'input_prompt', 'chain_of_thought_output'])
        for instruct_item in config['instruct']:
            schema_name = instruct_item['schema']
            if schema_name == 'MetastasisAssessment':
                schema = MetastasisAssessment
            elif schema_name == 'LNAssessment':
                schema = LNAssessment
            elif schema_name == 'TumorAssessment':
                schema = TumorAssessment
            elif schema_name == 'VascularAssessment':
                schema = VascularAssessment
            else:
                continue
            schema_keys = collect_model_fields(schema)
            full_log_keys.update(schema_keys)
        full_log_keys = sorted(full_log_keys)

        if not os.path.exists(full_log_csv):
            with open(full_log_csv, 'w', newline='', encoding='utf-8') as f_full:
                writer_full = csv.DictWriter(f_full, fieldnames=full_log_keys)
                writer_full.writeheader()

        for model_name in tqdm(config['models'], desc="Processing models"):
            logging.info(f"Processing model: {model_name}")

            if "gemma" in model_name.lower():
                quant_config = BitsAndBytesConfig(load_in_8bit=True)
                pipe = pipeline(
                    "text-generation",
                    model=model_name,
                    device_map='cuda',
                    torch_dtype=torch.bfloat16,
                    model_kwargs={
                        "quantization_config": quant_config,
                        "attn_implementation": "eager"
                    }
                )
            else:
                pipe = pipeline(
                    "text-generation",
                    model=model_name,
                    device_map='cuda',
                    torch_dtype="auto",
                    trust_remote_code=True,
                    model_kwargs={"attn_implementation": "flash_attention_2"}
                )

            outlines_tokenizer = outlines.models.TransformerTokenizer(
                AutoTokenizer.from_pretrained(model_name)
            )

            df_output = load_existing_output(output_csv, output_keys)
            data_per_report = {}
            full_log_results = []

            for instruct_item in config['instruct']:
                logging.info(f"Processing instruct: {instruct_item['name']}")

                schema_name = instruct_item['schema']
                if schema_name == 'MetastasisAssessment':
                    schema = MetastasisAssessment
                elif schema_name == 'LNAssessment':
                    schema = LNAssessment
                elif schema_name == 'TumorAssessment':
                    schema = TumorAssessment
                elif schema_name == 'VascularAssessment':
                    schema = VascularAssessment
                else:
                    continue

                logit = JSONLogitsProcessor(
                    schema=schema,
                    tokenizer=outlines_tokenizer,
                    whitespace_pattern=r"\s*"
                )

                instruct = instruct_item['prompt']
                collect_columns = instruct_item.get('collect_columns', [])
                results = process_dataset(dataset, instruct_item, pipe, logit)

                for example in results:
                    index = example.get('index')
                    if index not in data_per_report:
                        data_per_report[index] = {
                            'index': index,
                            'model_name': model_name
                        }
                        for col in all_collect_columns:
                            data_per_report[index][col] = ''

                    for col in collect_columns:
                        keys = col.split('.')
                        value = example['prediction']
                        for key in keys:
                            if isinstance(value, dict):
                                value = value.get(key, '')
                            else:
                                value = ''
                                break
                        data_per_report[index][col] = value

                    flat_example = {
                        'index': example.get('index', ''),
                        'model_name': model_name,
                        'instruct_name': instruct_item['name'],
                        'input_prompt': example.get('input_prompt', ''),
                        'chain_of_thought_output': example.get('chain_of_thought_output', '')
                    }
                    flattened_prediction = flatten_dict(example.get('prediction', {}))
                    flat_example.update(flattened_prediction)
                    full_log_results.append(flat_example)

                logging.info(f"Completed processing for instruct: {instruct_item['name']}")
                update_output_csv(df_output, data_per_report, output_csv)
                logging.info(f"Wrote output.csv for instruct: {instruct_item['name']}")

            with open(full_log_csv, 'a', newline='', encoding='utf-8') as f_full:
                writer_full = csv.DictWriter(f_full, fieldnames=full_log_keys, extrasaction='ignore')
                for row in full_log_results:
                    writer_full.writerow(row)

            data_per_report.clear()
            full_log_results.clear()

            del pipe
            torch.cuda.empty_cache()
            gc.collect()

        logging.info(f"Analysis complete. Results are logged in '{base_log_dir}'.")
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        sys.exit(1)
