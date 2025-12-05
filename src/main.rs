use fastembed::{TextEmbedding, InitOptions, EmbeddingModel};

fn main() -> anyhow::Result<()> {
    // 1. Initialize the Model
    // Fixed capitalization: AllMiniLML6V2
    let model = TextEmbedding::try_new(
        InitOptions::new(EmbeddingModel::AllMiniLML6V2)
            .with_show_download_progress(true)
            .with_cache_dir("model_cache".into())
    )?;

    // 2. The Input Data
    let anchor = "The dog barked";
    let sentences = vec![
        "The canine made noise", // Meaning match
        "The cat slept",         // No match - Rubblish
    ];

    // 3. Generate Vectors
    let anchor_vec = model.embed(vec![anchor], None)?[0].clone();
    let sentence_vecs = model.embed(sentences.clone(), None)?;

    println!("Comparing against: '{}'\n", anchor);

    // 4. Compare
    for (i, vec) in sentence_vecs.iter().enumerate() {
        // Dot Product
        let score: f32 = anchor_vec.iter().zip(vec).map(|(a, b)| a * b).sum();
        println!("Score: {:.2} | Sentence: {}", score, sentences[i]);
    }

    Ok(())
}