use fastembed::{TextEmbedding, InitOptions, EmbeddingModel};

fn main() -> anyhow::Result<()> {
    // 1. Load the Model (Downloads automatically on first run)
    // This forces the model to download into a folder named "model_cache" inside your project
    let model = TextEmbedding::try_new(
    InitOptions::new(EmbeddingModel::AllMiniLmL6V2)
        .with_cache_dir("model_cache".into()) 
    )?;

    // 2. The input data
    let anchor = "The dog barked";
    let sentences = vec![
        "The canine made noise", // Meaning match
        "The cat slept",         // No match
    ];

    // 3. Convert text to vectors
    // embed() takes a list, so we put the anchor in a vec![]
    let anchor_vec = &model.embed(vec![anchor], None)?[0];
    let sentence_vecs = model.embed(sentences.clone(), None)?;

    println!("Comparing against: '{}'\n", anchor);

    // 4. Compare
    for (i, vec) in sentence_vecs.iter().enumerate() {
        // Simple Dot Product: Multiply matching numbers and sum them up
        let score: f32 = anchor_vec.iter().zip(vec).map(|(a, b)| a * b).sum();
        
        println!("Score: {:.2} | Sentence: {}", score, sentences[i]);
    }

    Ok(())
}