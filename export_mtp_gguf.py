#!/usr/bin/env python3
"""
Merge trained MTP head weights into base GGUF model for llama.cpp (--spec-type draft-mtp).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "llama.cpp" / "gguf-py"))

import torch
from gguf import GGUFReader, GGUFWriter


def export_mtp_gguf(base_gguf_path: Path, mtp_weights_path: Path, output_gguf_path: Path):
    print(f"Loading Base GGUF: {base_gguf_path}")
    reader = GGUFReader(base_gguf_path, "r")
    
    print(f"Loading MTP Tensors: {mtp_weights_path}")
    mtp_tensors = torch.load(mtp_weights_path, map_location="cpu")

    writer = GGUFWriter(output_gguf_path, "nanbeige")

    # Copy base GGUF KV metadata & update MTP fields
    for key, field in reader.fields.items():
        if key == "general.architecture":
            continue

        value = field.contents()
        if key == "general.name":
            value = f"{value} MTP-1"

        subtype = field.types[-1] if field.types[0] == GGUFValueType.ARRAY else None
        writer.add_key_value(key, value, field.types[0], sub_type=subtype)

    # Add MTP metadata key
    writer.add_uint32("nanbeige.nextn_predict_layers", 1)

    # Copy base tensors
    for tensor in reader.tensors:
        writer.add_tensor(
            tensor.name,
            tensor.data,
            raw_dtype=tensor.tensor_type,
        )

    # Append trained MTP tensors
    print(f"Appending {len(mtp_tensors)} MTP tensors...")
    for name, param in mtp_tensors.items():
        param_np = param.detach().numpy().astype("float32")
        writer.add_tensor(name, param_np)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()
    print(f"Exported MTP GGUF model to: {output_gguf_path}")


def main():
    parser = argparse.ArgumentParser(description="Merge MTP weights into Nanbeige GGUF model")
    parser.add_argument("--base-gguf", type=Path, default=Path("../models/nanbeige4.2-3b-Q4_0.gguf"))
    parser.add_argument("--mtp-weights", type=Path, default=Path("./mtp_output/nanbeige_mtp_gguf_tensors.pt"))
    parser.add_argument("--output", type=Path, default=Path("../models/nanbeige4.2-3b-mtp-Q4_0.gguf"))
    args = parser.parse_args()

    export_mtp_gguf(args.base_gguf, args.mtp_weights, args.output)


if __name__ == "__main__":
    main()
