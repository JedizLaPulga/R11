use fastembed::{TextEmbedding, InitOptions, EmbeddingModel};
use std::io::{self, Write};

fn main() -> anyhow::Result<()> {
    // 1. Initialize the Model
    let model = TextEmbedding::try_new(
        InitOptions::new(EmbeddingModel::AllMiniLML6V2)
            .with_show_download_progress(true)
            .with_cache_dir("model_cache".into())
    )?;

    // 2. Define Mutable Topics (Converted to String for flexibility)
    // We use .to_string() so we can add new ones later.
    let mut topics: Vec<String> = vec![
        "Programming",
        "Cooking",
        "Politics",
        "Sports",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();

    println!("Loading Brain...");
    
    // We calculate embeddings immediately
    let mut topic_embeddings = model.embed(topics.clone(), None)?;

    println!("Ready! Type a sentence to classify.");
    println!("Commands:");
    println!("  exit      -> Quit");
    println!("  +Category -> Add a new topic (e.g. '+Mechanics')");
    println!("  ?list     -> See all topics\n");

    // 3. The Loop
    loop {
        print!("> ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        // --- COMMANDS ---

        if input.eq_ignore_ascii_case("exit") {
            break; // This breaks the loop, going to the Ok(()) below
        }
        
        if input.is_empty() {
            continue;
        }

        // Add Topic Command
        if input.starts_with('+') {
            let new_topic = &input[1..]; // Remove the '+'
            if !new_topic.is_empty() {
                // FIXED: We push a String, which matches our new Vec type
                topics.push(new_topic.to_string());
                
                // Recalculate embeddings to include the new topic
                println!("🧠 Learning '{}'...", new_topic);
                topic_embeddings = model.embed(topics.clone(), None)?;
                println!("✅ Learned!");
            }
            continue;
        }

        // List Command
        if input == "?list" {
            println!("Current Knowledge: {:?}", topics);
            continue;
        }

        // --- CLASSIFICATION LOGIC ---

        // Embed user input
        let input_vec = match model.embed(vec![input], None) {
            Ok(vecs) => vecs[0].clone(),
            Err(_) => { println!("Error processing text"); continue; }
        };

        // Find best match
        let mut best_score: f32 = -1.0;
        let mut best_topic_index = 0;

        for (i, topic_vec) in topic_embeddings.iter().enumerate() {
            // Dot Product
            let score: f32 = input_vec.iter().zip(topic_vec).map(|(a, b)| a * b).sum();
            
            if score > best_score {
                best_score = score;
                best_topic_index = i;
            }
        }

        // Output Result
        if best_score < 0.25 {
            println!("Topic: [Unknown]");
        } else {
            println!("Topic: {}", topics[best_topic_index]);
        }
        println!();
    }

    Ok(())
}