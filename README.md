[![Codespaces Prebuilds](https://github.com/nogibjj/assimilate-openai/actions/workflows/codespaces/create_codespaces_prebuilds/badge.svg)](https://github.com/nogibjj/assimilate-openai/actions/workflows/codespaces/create_codespaces_prebuilds)


Update of fork status, jcmeunier77code (july 8, 2025)
[![Python application test with Github Actions using devcontainers](https://github.com/jcmeunier77code/assimilate-openai/actions/workflows/main.yml/badge.svg)](https://github.com/jcmeunier77code/assimilate-openai/actions/workflows/main.yml)

# assimilate-openai

An extensive collection of tutorials and live streams on [OpenAi](https://openai.com/)

## Day 8:  Try to build a chatGTP client from scratch

Fixed it:

```bash
(.venv) @noahgift ➜ /workspaces/assimilate-openai/rust-curl-openai (main) $ cargo run
   Compiling reqwest v0.11.14
   Compiling rust-curl-openai v0.1.0 (/workspaces/assimilate-openai/rust-curl-openai)
    Finished dev [unoptimized + debuginfo] target(s) in 4.78s
     Running `target/debug/rust-curl-openai`
{"id":"cmpl-6rDd8mzOtMx7kKobqV0isiC7TkqU4","object":"text_completion","created":1678141798,"model":"text-davinci-003","choices":[{"text":"\n\nJupiter is the fifth planet from the Sun and the biggest one in our Solar System. It is very bright and can be seen in the night sky. It is named after the Roman god Jupiter. It is usually the third brightest thing you can see in the night sky after the Moon and Venus.","index":0,"logprobs":null,"finish_reason":"stop"}],"usage":{"prompt_tokens":151,"completion_tokens":62,"total_tokens":213}}
```

Partial progress here:  `rust-curl-openai`, but don't fully understand API yet.

A.  find a Rust library that does basic web requests like Python requests
```json
curl https://api.openai.com/v1/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
  "model": "text-davinci-003",
  "prompt": "Summarize this for a second-grade student:\n\nJupiter is the fifth planet from the Sun and the largest in the Solar System. It is a gas giant with a mass one-thousandth that of the Sun, but two-and-a-half times that of all the other planets in the Solar System combined. Jupiter is one of the brightest objects visible to the naked eye in the night sky, and has been known to ancient civilizations since before recorded history. It is named after the Roman god Jupiter.[19] When viewed from Earth, Jupiter can be bright enough for its reflected light to cast visible shadows,[20] and is on average the third-brightest natural object in the night sky after the Moon and Venus.",
  "temperature": 0.7,
  "max_tokens": 256,
  "top_p": 1,
  "frequency_penalty": 0,
  "presence_penalty": 0
}'
```


## Day 7:  More Rust

Ran into some problems with unfinished APIS.  Did find an example of chatgpt with Rust:  https://github.com/elbruno/RustOpenAIAPIs
Recap of the previous project which is in openai:

`cargo run -- complete -t "The weather in March is warmer than usual, this is because"`


## Day 6: Switching to Rust  

* install Rust via Rustup: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
Use Rust API for OpenAI (3rd party):  https://github.com/deontologician/openai-api-rust

* Create new project: `cargo new openai` and cd into it
* `make format` then `make lint` then `cargo run`

Working Example:

```bash
(.venv) @noahgift ➜ /workspaces/assimilate-openai/openai (main) $ cargo run -- complete -t "The rain in spain"
    Finished dev [unoptimized + debuginfo] target(s) in 0.14s
     Running `target/debug/openai complete -t 'The rain in spain'`
Completing: The rain in spain
Loves gets you nowhere
The rain in spain
```

`lib.rs`
```rust
/*This uses Open AI to Complete Sentences */

//accets the prompt and returns the completion
pub async fn complete_prompt(prompt: &str) -> Result<String, Box<dyn std::error::Error>> {
    let api_token = std::env::var("OPENAI_API_KEY")?;
    let client = openai_api::Client::new(&api_token);
    let prompt = String::from(prompt);
    let result = client.complete_prompt(prompt.as_str()).await?;
    Ok(result.choices[0].text.clone())
}

```


## Day 5:  Testing ChatGPT



## Day 4:  Setup prebuild for Codespaces and dive deep on Streamlit

`streamlit-apps`:  Using this as the starter.

* To run streamlit inside of Github codespaces do this:  `streamlit run streamlit/uberDemo.py --server.enableCORS=false`

## Day 3:  CI/CD and Refactor and built working streamlit app that generates code


* [how to generate code programatically](https://beta.openai.com/docs/guides/code)

## Day 2:  

* [Build Question and Answer CLI](https://github.com/nogibjj/assimilate-openai/blob/main/questionAnswerCLI.py)

## Day 1:  Exploring all of the concepts

* [YouTube](https://www.youtube.com/watch?v=lgGyDd_fQrA)
* [O'Reilly](https://learning.oreilly.com/videos/assimilate-openai/08252022VIDEOPAIML/08252022VIDEOPAIML-c1_s0/)

## References

* [twenty things with GPT-3](https://towardsdatascience.com/20-creative-things-to-try-out-with-gpt-3-2aacee3e2abf)
* [GPT-3 Use Cases](https://medium.com/eoraa-co/trending-use-cases-of-gpt-3-by-openai-56318b6a9184)
* [Top Three GPT-3](https://www.educative.io/blog/top-uses-gpt-3-deep-learning)
* [OpenAI cookbook](https://github.com/openai/openai-cookbook/tree/main/examples)
