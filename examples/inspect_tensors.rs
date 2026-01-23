//! Inspect tensor names and shapes in model files
use candle_core::{Device, Tensor};
use std::collections::HashMap;
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let device = Device::Cpu;

    // Main model - check code_predictor
    println!("=== Code predictor tensors ===");
    let main_path = Path::new("test_data/model/model.safetensors");
    if main_path.exists() {
        let tensors: HashMap<String, Tensor> = candle_core::safetensors::load(main_path, &device)?;
        let mut cp_tensors: Vec<_> = tensors
            .iter()
            .filter(|(k, _)| k.contains("code_predictor"))
            .collect();
        cp_tensors.sort_by_key(|(k, _)| k.clone());

        for (name, t) in &cp_tensors {
            println!("  {}: {:?}", name, t.dims());
        }
        println!("  ({} total)", cp_tensors.len());
    }

    // Speech tokenizer
    println!("\n=== Speech tokenizer tensors (sample) ===");
    let st_path = Path::new("test_data/speech_tokenizer/model.safetensors");
    if st_path.exists() {
        let tensors: HashMap<String, Tensor> = candle_core::safetensors::load(st_path, &device)?;
        let mut keys: Vec<_> = tensors.keys().collect();
        keys.sort();

        // Show quantizer tensors first
        println!("-- Quantizer --");
        let quant_keys: Vec<_> = keys.iter().filter(|k| k.contains("quantizer")).collect();
        for k in quant_keys.iter().take(20) {
            println!("  {}: {:?}", k, tensors[k.as_str()].dims());
        }
        println!("  ... ({} quantizer tensors)", quant_keys.len());

        // Show pre_transformer tensors
        println!("-- Pre-transformer --");
        let pre_keys: Vec<_> = keys
            .iter()
            .filter(|k| k.contains("pre_transformer") || k.contains("pre_conv"))
            .collect();
        for k in pre_keys.iter().take(15) {
            println!("  {}: {:?}", k, tensors[k.as_str()].dims());
        }
        println!("  ... ({} pre_transformer tensors)", pre_keys.len());

        // Show upsample tensors
        println!("-- Upsample --");
        let up_keys: Vec<_> = keys.iter().filter(|k| k.contains("upsample")).collect();
        for k in up_keys.iter().take(10) {
            println!("  {}: {:?}", k, tensors[k.as_str()].dims());
        }
        println!("  ... ({} upsample tensors)", up_keys.len());

        println!("  ({} total tensors)", keys.len());

        // Show input/output proj tensors
        println!("-- Input/Output projections --");
        let proj_keys: Vec<_> = keys
            .iter()
            .filter(|k| k.contains("input_proj") || k.contains("output_proj"))
            .collect();
        for k in proj_keys.iter().take(10) {
            println!("  {}: {:?}", k, tensors[k.as_str()].dims());
        }
    }

    Ok(())
}
