use fastembed::{TextEmbedding, InitOptions, EmbeddingModel};
use std::io::{self, Write};

fn main() -> anyhow::Result<()> {
    println!("Loading Brain Model... (This happens once)");

    // 1. Initialize the Model
    let model = TextEmbedding::try_new(
        InitOptions::new(EmbeddingModel::AllMiniLML6V2)
            .with_show_download_progress(true)
            .with_cache_dir("model_cache".into())
    )?;

    // 2. Define the "Concepts" (The AI's vocabulary for topics)
    // You can add as many as you want here.
    let topics = vec![
        "Programming & Coding",
        "Science & Nature",
        "Politics & News",
        "Food & Cooking",
        "Movies & Music",
        "Sports & Fitness",
        "Business & Finance",
        "Love & Relationships",
        "Travel & Adventure",
        "Casual Greeting",
    ];

    // 3. Pre-calculate Topic Vectors (Optimization)
    // We calculate these ONCE so the loop is instant.
    let topic_embeddings = model.embed(topics.clone(), None)?;

    println!("ready. Type 'exit' to quit.\n");

    // 4. The Loop
    loop {
        // Print prompt without new line
        print!("User Input: ");
        io::stdout().flush()?; 

        // Read input
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        // Exit condition
        if input.eq_ignore_ascii_case("exit") {
            break;
        }
        if input.is_empty() {
            continue;
        }

        // 5. The Magic: Classify the Input
        // Convert user text to vector
        let input_vec = match model.embed(vec![input], None) {
            Ok(vecs) => vecs[0].clone(),
            Err(_) => { println!("Error processing text"); continue; }
        };

        // Find the closest topic
        let mut best_score: f32 = -1.0;
        let mut best_topic_index = 0;

        for (i, topic_vec) in topic_embeddings.iter().enumerate() {
            // Calculate Similarity (Dot Product)
            let score: f32 = input_vec.iter().zip(topic_vec).map(|(a, b)| a * b).sum();
            
            if score > best_score {
                best_score = score;
                best_topic_index = i;
            }
        }

        // 6. Output Result
        // We add a threshold: if similarity is too low, it's just noise.
        if best_score < 0.25 {
            println!("AI Topic:   [Unclear / Random]");
        } else {
            println!("AI Topic:   {}", topics[best_topic_index]);
        }
        println!(); // Spacing
    }

    Ok(())
}