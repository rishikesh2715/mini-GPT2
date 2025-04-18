def test_imports():
    print("✅ Testing essential imports...")
    try:
        import pandas, numpy, matplotlib, scipy, sklearn, transformers, torch, tqdm, datasets
        print("✅ All libraries imported successfully.")
    except ImportError as e:
        print("❌ Import failed:", e)
        return False
    return True


def test_cuda():
    print("\n⚙️ Checking PyTorch and CUDA availability...")
    import torch
    print(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"✅ CUDA is available. GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ CUDA not available. Using CPU.")


def test_transformers_pipeline():
    print("\n🧠 Testing transformers pipeline (GPT-2)...")
    try:
        from transformers import pipeline
        gen = pipeline("text-generation", model="gpt2")
        output = gen("The universe is", max_length=10)
        print("✅ Transformers pipeline works.")
        print("🔍 Sample output:", output[0]['generated_text'])
    except Exception as e:
        print("❌ Transformers pipeline failed:", e)


if __name__ == "__main__":
    if test_imports():
        test_cuda()
        test_transformers_pipeline()
