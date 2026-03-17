## llama.cpp

llama.cpp is an open-source C++ library for running Large Language Models (LLMs) like Meta's LLaMA. Its primary goal is to make it possible to run these powerful AI models efficiently on everyday consumer hardware, like laptops, desktops, and even smartphones, without needing expensive GPUs.

### How Does GGML Relate to llama.cpp?
``` text

┌─────────────────────────────────────────────┐
│            Application Layer                │
│   (Ollama, LM Studio, GPT4All, Jan, etc.)  │
├─────────────────────────────────────────────┤
│              llama.cpp                      │
│   (LLM-specific logic: tokenization,       │
│    sampling, model loading, chat, etc.)     │
├─────────────────────────────────────────────┤
│                GGML                         │
│   (Low-level tensor library: matrix math,  │
│    quantization, memory management,        │
│    hardware-specific optimizations)         │
├─────────────────────────────────────────────┤
│              Hardware                       │
│   (CPU / GPU / Apple Metal / CUDA / etc.)   │
└─────────────────────────────────────────────┘
```
**GGML** = The math engine (general-purpose tensor operations)  
**llama.cpp** = The LLM application built on top of GGML

---

### The GGML File Format (Legacy) vs. GGUF
### GGML Format (Old)
The original file format used to store quantized models.  
**Had limitations:**  
- Model metadata was not self-contained.
- Adding new features required breaking changes.
- Tokenizer info had to be loaded separately.

### GGUF Format (Current)   
- Replaced the GGML format as the standard.
- Stands for GPT-Generated Unified Format.

**Improvements:**    
✅ Self-contained (model weights + tokenizer + metadata all in one file)  
✅ Extensible key-value metadata system  
✅ Forward and backward compatible  
✅ Single file — easy to distribute and use   


---

## How does llama.cpp work?

llama.cpp is a C/C++ implementation for running Meta's LLaMA (and other) large language models efficiently on consumer hardware, primarily on CPUs (with optional GPU offloading). Created by Georgi Gerganov, it focuses on performance and minimal dependencies.

### Core Architecture
**1. Model Loading**   
Reads model weights from a GGUF file format (previously GGML)

GGUF is a binary format that stores:
- Model hyperparameters (dimensions, number of layers, vocab size, etc.)
- Tokenizer data (vocabulary, merge rules)
- Quantized or full-precision weight tensors
- Weights are memory-mapped (mmap) for efficient loading without reading everything into RAM at once

---

**2. Quantization**    
This is one of the key innovations. Instead of using full float32 (or float16) weights:
| Format | Bits per weight | Description                  |
|--------|-----------------|------------------------------|
| Q2_K   | ~2.5 bits       | Extremely compressed         |
| Q4_0   | 4 bits          | Basic 4-bit quantization     |
| Q4_K_M | ~4.5 bits       | K-quant medium quality       |
| Q5_K_M | ~5.5 bits       | Good balance                 |
| Q6_K   | ~6.5 bits       | Near-original quality        |
| Q8_0   | 8 bits          | High quality                 |
| F16    | 16 bits         | Half precision               |


**How quantization works:**   

- Weights are grouped into blocks (e.g., 32 values)
- Each block stores a scale factor (and sometimes a zero point) in higher precision
- Individual weights are stored as small integers
- During inference, weights are dequantized on-the-fly: w = scale * quantized_value

---

**3. ggml Tensor Library**  
llama.cpp is built on top of ggml, a custom tensor library (also by Gerganov):

``` text

ggml (tensor library)
  ├── Tensor operations (matmul, softmax, RoPE, etc.)
  ├── Computation graph construction
  ├── Memory management (scratch buffers, memory pools)
  └── Backend dispatching (CPU, CUDA, Metal, Vulkan, etc.)

```  
**Key aspects:**   
- Static computation graph: The forward pass is built as a DAG of operations, then executed
- No automatic differentiation (inference only, no training)
- Manual memory management with arena allocators for efficiency

---

## Inference Pipeline
**Step-by-step token generation:**   

```
Input text
    │
    ▼
┌──────────────┐
│  Tokenizer   │  (BPE/SentencePiece → token IDs)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Embedding   │  token IDs → embedding vectors
└──────┬───────┘
       │
       ▼
┌──────────────────────────────┐
│  Transformer Layers (×N)     │
│  ┌────────────────────────┐  │
│  │ RMSNorm                │  │
│  │ Self-Attention (+ RoPE)│  │  ← KV Cache used here
│  │ RMSNorm                │  │
│  │ Feed-Forward (SwiGLU)  │  │
│  │ Residual connections   │  │
│  └────────────────────────┘  │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────┐
│  Final Norm  │
│  + Linear    │  → logits (vocab-sized vector)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Sampling    │  (temperature, top-k, top-p, etc.)
└──────┬───────┘
       │
       ▼
  Output token → detokenize → text
  (feed back for next token)

``` 
---

### App Flow

```
┌─────────────────────────────────────────┐
│        Android App (Kotlin/Java)        │
│  ┌───────────────────────────────────┐  │
│  │    UI Layer (Activity/Compose)    │  │
│  └─────────────┬─────────────────────┘  │
│                │                         │
│  ┌─────────────▼─────────────────────┐  │
│  │   ViewModel / Business Logic      │  │
│  └─────────────┬─────────────────────┘  │
│                │                         │
│  ┌─────────────▼─────────────────────┐  │
│  │   JNI Wrapper (Kotlin/Java)       │  │
│  │   - Load native library           │  │
│  │   - Define native methods         │  │
│  └─────────────┬─────────────────────┘  │
└────────────────┼─────────────────────────┘
                 │ JNI Boundary
┌────────────────▼─────────────────────────┐
│         Native Layer (C/C++)             │
│  ┌───────────────────────────────────┐  │
│  │   JNI Bridge Implementation       │  │
│  │   - Convert Java ↔ C++ types      │  │
│  └─────────────┬─────────────────────┘  │
│                │                         │
│  ┌─────────────▼─────────────────────┐  │
│  │      llama.cpp Core               │  │
│  │   - Model loading                 │  │
│  │   - Inference engine              │  │
│  │   - Tokenization                  │  │
│  └───────────────────────────────────┘  │
└──────────────────────────────────────────┘
                 │
                 ▼
         ┌──────────────┐
         │  Model File  │
         │  (.gguf)     │
         └──────────────┘
```

### Integration Methods
**Method 1:** Using Pre-built Library (Recommended for beginners)   
Several projects provide ready-to-use Android bindings: llama.cpp's official Android example: examples/llama.android/

**Method 2:** Building from Source (Full control)   
Build llama.cpp as an Android native library using CMake and NDK.
Step-by-Step Integration (Method 2 - From Source)

**a.** Project Structure:   

```text

YourAndroidProject/
├── app/
│   ├── src/
│   │   ├── main/
│   │   │   ├── java/com/example/llama/
│   │   │   │   ├── MainActivity.kt
│   │   │   │   └── LlamaAndroid.kt  (JNI wrapper)
│   │   │   ├── cpp/
│   │   │   │   ├── llama.cpp/       (git submodule or copy)
│   │   │   │   ├── jni_bridge.cpp   (your JNI code)
│   │   │   │   └── CMakeLists.txt
│   │   │   ├── assets/
│   │   │   │   └── models/
│   │   │   │       └── model.gguf   (your quantized model)
│   │   │   └── AndroidManifest.xml
│   │   └── build.gradle.kts
│   └── build.gradle.kts
└── build.gradle.kts
```

**b.** CMake Configuration   
app/src/main/cpp/CMakeLists.txt:

```cmake

cmake_minimum_required(VERSION 3.22.1)
project(llama_android)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add llama.cpp library
add_subdirectory(llama.cpp)

# Create our JNI bridge library
add_library(llama_android SHARED
    jni_bridge.cpp
)

# Link against llama.cpp
target_link_libraries(llama_android
    llama        # Main llama.cpp library
    common       # Common utilities from llama.cpp
    log          # Android logging
    android      # Android native API
)

# Include directories
target_include_directories(llama_android PRIVATE
    llama.cpp/include
    llama.cpp/common
    llama.cpp/ggml/include
)
```

**c.** JNI Bridge (C++ Side)   
app/src/main/cpp/jni_bridge.cpp:

```C++

#include <jni.h>
#include <android/log.h>
#include <string>
#include <thread>
#include "llama.h"
#include "common.h"

#define LOG_TAG "LlamaCpp"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// Global context holders
static llama_model* g_model = nullptr;
static llama_context* g_ctx = nullptr;
static llama_sampler* g_sampler = nullptr;

extern "C" {

// Load model
JNIEXPORT jlong JNICALL
Java_com_example_llama_LlamaAndroid_loadModel(
    JNIEnv* env,
    jobject /* this */,
    jstring modelPath,
    jint nCtx,
    jint nThreads
) {
    const char* model_path_cstr = env->GetStringUTFChars(modelPath, nullptr);
    LOGD("Loading model from: %s", model_path_cstr);
    
    // Initialize backend
    llama_backend_init();
    llama_numa_init(GGML_NUMA_STRATEGY_DISABLED);
    
    // Model parameters
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;  // CPU only on mobile
    
    // Load model
    g_model = llama_load_model_from_file(model_path_cstr, model_params);
    env->ReleaseStringUTFChars(modelPath, model_path_cstr);
    
    if (!g_model) {
        LOGE("Failed to load model");
        return 0;
    }
    
    // Context parameters
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = nCtx;
    ctx_params.n_threads = nThreads;
    ctx_params.n_threads_batch = nThreads;
    
    // Create context
    g_ctx = llama_new_context_with_model(g_model, ctx_params);
    if (!g_ctx) {
        LOGE("Failed to create context");
        llama_free_model(g_model);
        return 0;
    }
    
    // Create sampler
    auto sparams = llama_sampler_chain_default_params();
    g_sampler = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(g_sampler, 
        llama_sampler_init_temp(0.8f));
    llama_sampler_chain_add(g_sampler, 
        llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
    
    LOGD("Model loaded successfully");
    return reinterpret_cast<jlong>(g_ctx);
}

// Tokenize text
JNIEXPORT jintArray JNICALL
Java_com_example_llama_LlamaAndroid_tokenize(
    JNIEnv* env,
    jobject /* this */,
    jstring text,
    jboolean addBos
) {
    if (!g_model) return nullptr;
    
    const char* text_cstr = env->GetStringUTFChars(text, nullptr);
    
    // Tokenize
    std::vector<llama_token> tokens;
    tokens.resize(strlen(text_cstr) + (addBos ? 1 : 0));
    
    int n_tokens = llama_tokenize(
        g_model,
        text_cstr,
        strlen(text_cstr),
        tokens.data(),
        tokens.size(),
        addBos,
        false  // special tokens
    );
    
    env->ReleaseStringUTFChars(text, text_cstr);
    
    if (n_tokens < 0) {
        tokens.resize(-n_tokens);
        llama_tokenize(g_model, text_cstr, strlen(text_cstr),
                      tokens.data(), tokens.size(), addBos, false);
        n_tokens = tokens.size();
    } else {
        tokens.resize(n_tokens);
    }
    
    // Convert to Java int array
    jintArray result = env->NewIntArray(n_tokens);
    env->SetIntArrayRegion(result, 0, n_tokens, 
                          reinterpret_cast<jint*>(tokens.data()));
    return result;
}

// Generate completion
JNIEXPORT jstring JNICALL
Java_com_example_llama_LlamaAndroid_generate(
    JNIEnv* env,
    jobject /* this */,
    jstring prompt,
    jint maxTokens,
    jobject callback  // for streaming
) {
    if (!g_ctx || !g_model) return env->NewStringUTF("");
    
    const char* prompt_cstr = env->GetStringUTFChars(prompt, nullptr);
    
    // Tokenize prompt
    std::vector<llama_token> tokens;
    tokens.resize(strlen(prompt_cstr) + 1);
    int n_tokens = llama_tokenize(g_model, prompt_cstr, 
                                  strlen(prompt_cstr),
                                  tokens.data(), tokens.size(), 
                                  true, false);
    tokens.resize(n_tokens);
    
    env->ReleaseStringUTFChars(prompt, prompt_cstr);
    
    // Clear KV cache
    llama_kv_cache_clear(g_ctx);
    
    // Evaluate prompt
    llama_batch batch = llama_batch_init(tokens.size(), 0, 1);
    for (size_t i = 0; i < tokens.size(); i++) {
        llama_batch_add(batch, tokens[i], i, {0}, false);
    }
    batch.logits[batch.n_tokens - 1] = true;  // only need last logit
    
    if (llama_decode(g_ctx, batch) != 0) {
        LOGE("Failed to decode prompt");
        llama_batch_free(batch);
        return env->NewStringUTF("");
    }
    
    std::string result;
    
    // Generate tokens
    for (int i = 0; i < maxTokens; i++) {
        // Sample next token
        llama_token new_token = llama_sampler_sample(g_sampler, g_ctx, -1);
        
        // Check for EOS
        if (llama_token_is_eog(g_model, new_token)) {
            break;
        }
        
        // Decode token to text
        char buf[256];
        int n = llama_token_to_piece(g_model, new_token, buf, sizeof(buf), 0, true);
        if (n > 0) {
            result.append(buf, n);
            
            // Stream callback (optional)
            if (callback) {
                jclass callbackClass = env->GetObjectClass(callback);
                jmethodID onToken = env->GetMethodID(callbackClass, 
                    "onToken", "(Ljava/lang/String;)V");
                if (onToken) {
                    jstring token_str = env->NewStringUTF(std::string(buf, n).c_str());
                    env->CallVoidMethod(callback, onToken, token_str);
                    env->DeleteLocalRef(token_str);
                }
            }
        }
        
        // Prepare for next iteration
        llama_batch_clear(batch);
        llama_batch_add(batch, new_token, tokens.size() + i, {0}, true);
        
        if (llama_decode(g_ctx, batch) != 0) {
            LOGE("Failed to decode");
            break;
        }
    }
    
    llama_batch_free(batch);
    return env->NewStringUTF(result.c_str());
}

// Free resources
JNIEXPORT void JNICALL
Java_com_example_llama_LlamaAndroid_freeModel(JNIEnv* env, jobject /* this */) {
    if (g_sampler) {
        llama_sampler_free(g_sampler);
        g_sampler = nullptr;
    }
    if (g_ctx) {
        llama_free(g_ctx);
        g_ctx = nullptr;
    }
    if (g_model) {
        llama_free_model(g_model);
        g_model = nullptr;
    }
    llama_backend_free();
    LOGD("Model freed");
}

} // extern "C"
```

**d.** Kotlin/Java Wrapper
LlamaAndroid.kt:

```Kotlin

package com.example.llama

import android.content.Context
import java.io.File

class LlamaAndroid(private val context: Context) {
    
    companion object {
        init {
            System.loadLibrary("llama_android")
        }
    }
    
    // Native methods
    private external fun loadModel(
        modelPath: String, 
        nCtx: Int, 
        nThreads: Int
    ): Long
    
    private external fun tokenize(
        text: String, 
        addBos: Boolean
    ): IntArray
    
    private external fun generate(
        prompt: String, 
        maxTokens: Int,
        callback: TokenCallback?
    ): String
    
    external fun freeModel()
    
    // Callback interface for streaming
    interface TokenCallback {
        fun onToken(token: String)
    }
    
    private var isLoaded = false
    
    // Load model from assets or file
    fun load(modelFileName: String, nCtx: Int = 2048): Boolean {
        val modelPath = getModelPath(modelFileName)
        if (modelPath == null) {
            return false
        }
        
        val nThreads = Runtime.getRuntime().availableProcessors()
        val result = loadModel(modelPath, nCtx, nThreads)
        isLoaded = result != 0L
        return isLoaded
    }
    
    // Get tokens from text
    fun getTokens(text: String, addBos: Boolean = true): IntArray {
        if (!isLoaded) return intArrayOf()
        return tokenize(text, addBos)
    }
    
    // Generate completion (blocking)
    fun complete(prompt: String, maxTokens: Int = 512): String {
        if (!isLoaded) return ""
        return generate(prompt, maxTokens, null)
    }
    
    // Generate with streaming callback
    fun completeStreaming(
        prompt: String, 
        maxTokens: Int = 512,
        onToken: (String) -> Unit
    ): String {
        if (!isLoaded) return ""
        
        val callback = object : TokenCallback {
            override fun onToken(token: String) {
                onToken(token)
            }
        }
        
        return generate(prompt, maxTokens, callback)
    }
    
    // Helper to get model path
    private fun getModelPath(fileName: String): String? {
        // Try internal storage first
        val internalFile = File(context.filesDir, "models/$fileName")
        if (internalFile.exists()) {
            return internalFile.absolutePath
        }
        
        // Try to copy from assets
        return try {
            val outFile = File(context.filesDir, "models")
            outFile.mkdirs()
            val outputFile = File(outFile, fileName)
            
            context.assets.open("models/$fileName").use { input ->
                outputFile.outputStream().use { output ->
                    input.copyTo(output)
                }
            }
            outputFile.absolutePath
        } catch (e: Exception) {
            e.printStackTrace()
            null
        }
    }
    
    fun cleanup() {
        if (isLoaded) {
            freeModel()
            isLoaded = false
        }
    }
}
```

---

## Preparing SmolVLM 500M for llama.cpp
SmolVLM is a vision-language model that can process both images and text. Here's how to prepare and use it with llama.cpp on Android.

### Understanding SmolVLM Architecture
```
┌──────────────────────────────────────────────────┐
│              SmolVLM 500M                        │
├──────────────────────────────────────────────────┤
│                                                  │
│  ┌────────────────┐      ┌──────────────────┐   │
│  │  Image Input   │      │   Text Input     │   │
│  │  (e.g. 384x384)│      │   (Prompt)       │   │
│  └───────┬────────┘      └────────┬─────────┘   │
│          │                        │             │
│  ┌───────▼────────┐              │             │
│  │ Vision Encoder │              │             │
│  │  (SigLIP)      │              │             │
│  │  - Patch embed │              │             │
│  │  - Transformer │              │             │
│  └───────┬────────┘              │             │
│          │                        │             │
│  ┌───────▼────────┐              │             │
│  │  Projector     │              │             │
│  │  (MLP)         │              │             │
│  └───────┬────────┘              │             │
│          │                        │             │
│          └────────┬───────────────┘             │
│                   │                             │
│           ┌───────▼────────┐                    │
│           │ Language Model │                    │
│           │   (Phi-like)   │                    │
│           │  - Processes   │                    │
│           │    vision +    │                    │
│           │    text tokens │                    │
│           └───────┬────────┘                    │
│                   │                             │
│           ┌───────▼────────┐                    │
│           │  Text Output   │                    │
│           └────────────────┘                    │
└──────────────────────────────────────────────────┘

```
### App flow 
```text
User selects image
        │
        ▼
┌──────────────────────────────┐
│ Load image from gallery      │
│ Resize to 384x384            │
│ Convert to RGB               │
└───────────┬──────────────────┘
            │
            ▼
┌──────────────────────────────┐
│ Pass to native layer via JNI │
│ - Bitmap → uint8_t* pixels   │
└───────────┬──────────────────┘
            │
            ▼
┌──────────────────────────────┐
│ Vision Encoder (CLIP/SigLIP) │
│ 1. Split into patches        │
│ 2. Embed patches             │
│ 3. Run vision transformer    │
│ 4. Output: image embeddings  │
│    (e.g., 576 vectors)       │
└───────────┬──────────────────┘
            │
            ▼
┌──────────────────────────────┐
│ Projector (MLP)              │
│ - Map vision → text space    │
│ - Output: projected embeddings│
└───────────┬──────────────────┘
            │
            ▼
┌──────────────────────────────┐
│ Combine with text prompt     │
│ Format: <image> + question   │
│ Tokenize text parts          │
└───────────┬──────────────────┘
            │
            ▼
┌──────────────────────────────┐
│ Build batch:                 │
│ [img_emb1, img_emb2, ...,   │
│  text_tok1, text_tok2, ...]  │
└───────────┬──────────────────┘
            │
            ▼
┌──────────────────────────────┐
│ Language Model               │
│ - Process multimodal input   │
│ - Generate answer            │
└───────────┬──────────────────┘
            │
            ▼
┌──────────────────────────────┐
│ Return to Kotlin/UI          │
│ Display answer               │
└──────────────────────────────┘
```

---

### How to convert to gguf
https://github.com/ggml-org/llama.cpp/blob/master/docs/development/HOWTO-add-model.md

### Quantize model Steps:
https://github.com/ggml-org/llama.cpp/tree/master/tools/quantize
