import os
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
import chromadb
# from chromadb.config import Settings # Settings is often part of client setup, not strictly needed here for basic usage
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor # Assuming janus is correctly installed
from janus.utils.io import load_pil_images
import re # For parsing numbers from text

# --- 0. Helper function to parse numbers from LLM text output ---
def _extract_number_from_text(text: str) -> int:
    """
    Extracts a number from a text string.
    Handles digits and common number words.
    Returns 0 if no clear number is found.
    """
    text = text.lower()
    # Check for explicit "no" or "zero"
    if "no " in text or "zero" in text or "none" in text:
        return 0

    # Try to find digits
    digits = re.findall(r'\d+', text)
    if digits:
        return int(digits[0]) # Return the first number found

    # Check for number words (can be expanded)
    number_words = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "a single": 1, "a couple": 2, "several": 3 # Approximate for "several"
    }
    for word, num in number_words.items():
        if word in text:
            return num
            
    # Fallback if no number is clearly identified
    # This part is tricky, as the LLM might say "An ambulance is present"
    # For now, if it doesn't explicitly state zero or a number, and it mentions the vehicle,
    # we could assume at least one, but this can be inaccurate.
    # For a safer approach, default to 0 if no specific count is found.
    # If the prompt asks "How many..." and the answer is just "Yes", it implies at least one.
    # However, the current prompts are "How many...", so it should ideally provide a number.
    return 0


# --- 0.5 Helper function to call Janus model ---
def _call_janus_model(vl_gpt, vl_chat_processor, tokenizer, image_path, prompt_text, max_new_tokens=100):
    """
    Helper function to make a call to the Janus vision-language model.
    """
    conversation = [
        {
            "role": "<|User|>",
            "content": f"<image_placeholder>\n{prompt_text}",
            "images": [image_path],
        },
        {"role": "<|Assistant|>", "content": ""},
    ]
    
    pil_images = load_pil_images(conversation)
    prepare_inputs = vl_chat_processor(
        conversations=conversation, images=pil_images, force_batchify=True
    ).to(vl_gpt.device)
    
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
    
    outputs = vl_gpt.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
    )
    
    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    return answer.strip()

# --- 1. 登录 Hugging Face ---
def setup_hf():
    """
    Placeholder for Hugging Face setup.
    Ensure you are logged in (e.g., `huggingface-cli login`) or have HF_TOKEN set
    if your model requires authentication.
    """
    # Example: os.environ['HF_TOKEN'] = 'your_hf_token_here'
    # Or use: from huggingface_hub import login; login()
    print("Hugging Face setup: Assuming logged in or token is set via environment.")
    pass

# --- 2. 加载 Janus-Pro-7B 视觉模型 ---
def load_vision_model(model_id: str = "deepseek-ai/Janus-Pro-7B"):
    print(f"Loading vision model: {model_id}...")
    vl_chat_processor = VLChatProcessor.from_pretrained(model_id)
    tokenizer = vl_chat_processor.tokenizer
    
    vl_gpt = AutoModelForCausalLM.from_pretrained(
        model_id, 
        trust_remote_code=True
    )
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()
    print("Vision model loaded successfully.")
    return vl_gpt, vl_chat_processor, tokenizer

# --- 3. 初始化文本 Embedding 模型（用于 Chroma 检索） ---
def load_embed_model(model_name: str = 'all-MiniLM-L6-v2'):
    print(f"Loading embedding model: {model_name}...")
    model = SentenceTransformer(model_name)
    print("Embedding model loaded successfully.")
    return model

# --- 4. 初始化 Chroma 本地向量数据库 ---
def setup_chroma(db_path: str = "c:/users/admin/desktop/chroma_db", collection_name: str = "your_collection_name"):
    print(f"Setting up ChromaDB at: {db_path} with collection: {collection_name}")
    # client = chromadb.PersistentClient(path=db_path, settings=Settings(anonymized_telemetry=False)) # Example with settings
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(collection_name)
    print("ChromaDB setup complete.")
    return collection

# --- 5. 构建知识库（仅需运行一次，如果已构建可注释掉） ---
def build_knowledge_base(embed_model, collection):
    # List 1: Accident Descriptions (truncated for brevity in this example)
    accident_descriptions = [ 
        "A truck lies overturned on the highway median, surrounded by emergency vehicles and scattered debris.",
        "A red sedan has collided with a guardrail, its hood crumpled and brake lights illuminated.",
        "Two semi-trailers are locked together in a T-bone collision, blocking all lanes of traffic.",
        "A motorcycle rider lies motionless beside their bike after skidding off the wet road.",
        "Flames and smoke rise from a burning vehicle near an exit ramp, with firefighters on scene.",
        # ... (keep your full list)
    ]
    # List 2: Heavy Traffic Descriptions (Not Accidents)
    heavy_traffic_descriptions = [
        "Heavy traffic with vehicles moving slowly but orderly.",
        "Congested highway with bumper-to-bumper traffic.",
        "Traffic jam with vehicles at a complete standstill.",
        "Slow-moving traffic",
        "Congested roadway with no accidents or incidents visible.",
        "Vehicles waiting",
        "Heavy but orderly traffic on a city street."
    ]
    
    # List 3: Fast/Normal Traffic Descriptions
    fast_traffic_descriptions = [
        "Vehicles cruise smoothly on a wide, open multi-lane highway.",
        "Cars maintain consistent spacing while travelling quickly along a coastal road.",
        # ... (keep your full list)
    ]

    all_texts = accident_descriptions + heavy_traffic_descriptions + fast_traffic_descriptions
    all_ids = [f"desc_{i}" for i in range(len(all_texts))]
    all_metadatas = (
        [{"is_accident": True, "source": "accident_kb"} for _ in accident_descriptions] +
        [{"is_accident": False, "source": "heavy_traffic_kb"} for _ in heavy_traffic_descriptions] +
        [{"is_accident": False, "source": "fast_traffic_kb"} for _ in fast_traffic_descriptions]
    )

    print(f"Embedding {len(all_texts)} texts for knowledge base...")
    embeddings = embed_model.encode(all_texts, show_progress_bar=True).tolist()
    collection.upsert(
        ids=all_ids,
        embeddings=embeddings,
        documents=all_texts,
        metadatas=all_metadatas
    )
    print("Knowledge base built with", collection.count(), "entries.")

# --- 6. 使用 Janus 模型生成图像描述 ---
def generate_image_description(vl_gpt, vl_chat_processor, tokenizer, image_path):
    prompt = "Describe the image in one sentence."
    return _call_janus_model(vl_gpt, vl_chat_processor, tokenizer, image_path, prompt, max_new_tokens=100)

# --- 7. 使用 Janus 模型进行 RAG 推理 (Accident Detection) ---
def process_with_rag_accident(vl_gpt, vl_chat_processor, tokenizer, image_path, caption, retrieved_context):
    rag_prompt = f"Image description: {caption}\n"
    rag_prompt += "Retrieved knowledge:\n"
    for i, doc in enumerate(retrieved_context, start=1):
        rag_prompt += f"{i}. {doc}\n"
    rag_prompt += "Based on the image description and retrieved knowledge, is there a traffic accident in the image? Respond solely with 'yes' or 'no'."
    
    answer = _call_janus_model(vl_gpt, vl_chat_processor, tokenizer, image_path, rag_prompt, max_new_tokens=10) # yes/no is short
    
    answer = answer.lower().strip().replace(".", "")
    if "yes" in answer:
        return "yes"
    elif "no" in answer:
        return "no"
    else:
        # If the answer is not clearly yes or no, it might be better to log it or default to 'no'
        print(f"  [RAG] Ambiguous accident response: '{answer}'. Defaulting to 'no'.")
        return "no" 

# --- Functions to count emergency vehicles ---
def count_vehicles_in_image(vl_gpt, vl_chat_processor, tokenizer, image_path, vehicle_type_plural: str, vehicle_type_singular: str):
    """Generic function to count a type of vehicle."""
    prompt = f"How many {vehicle_type_plural} are visible in this image? If none, say 'zero {vehicle_type_plural}' or 'no {vehicle_type_plural}'."
    answer_text = _call_janus_model(vl_gpt, vl_chat_processor, tokenizer, image_path, prompt, max_new_tokens=50)
    count = _extract_number_from_text(answer_text)
    print(f"   Raw response for {vehicle_type_plural}: '{answer_text}', Extracted count: {count}")
    return count

def count_ambulances(vl_gpt, vl_chat_processor, tokenizer, image_path):
    return count_vehicles_in_image(vl_gpt, vl_chat_processor, tokenizer, image_path, "ambulances", "ambulance")

def count_firetrucks(vl_gpt, vl_chat_processor, tokenizer, image_path):
    return count_vehicles_in_image(vl_gpt, vl_chat_processor, tokenizer, image_path, "firetrucks", "firetruck")

def count_police_cars(vl_gpt, vl_chat_processor, tokenizer, image_path):
    return count_vehicles_in_image(vl_gpt, vl_chat_processor, tokenizer, image_path, "police cars", "police car")


# --- 8. 图像批量处理 + RAG 推理 ---
def run_inference(vl_gpt, vl_chat_processor, tokenizer, embed_model, collection, photo_dir):
    accident_ids = []
    ambulance_image_ids = [] # Images containing one or more ambulances
    firetruck_image_ids = [] # Images containing one or more firetrucks
    police_car_image_ids = []  # Images containing one or more police cars

    print("-" * 30)
    print(f"Processing directory: {photo_dir}")

    if not os.path.isdir(photo_dir):
        print(f"Error: Directory not found: {photo_dir}")
        return [], [], [], []

    try:
        image_files = [f for f in os.listdir(photo_dir)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]
    except OSError as e:
        print(f"Error accessing directory {photo_dir}: {e}")
        return [], [], [], []

    if not image_files:
        print("No image files found in this directory.")
        return [], [], [], []

    print(f"Found {len(image_files)} images to process.")

    for image_file in image_files:
        image_path = os.path.join(photo_dir, image_file)
        # image_id = os.path.splitext(image_file)[0] # Not strictly used later, path is identifier

        print(f"\nProcessing image: {image_file}")

        try:
            caption = generate_image_description(vl_gpt, vl_chat_processor, tokenizer, image_path)
            print(f"  Generated Caption: {caption}")

            # Count emergency vehicles
            num_ambulances = count_ambulances(vl_gpt, vl_chat_processor, tokenizer, image_path)
            if num_ambulances > 0:
                ambulance_image_ids.append(image_path)
                print(f"  >>>> Ambulance(s) detected: {num_ambulances}. Added to list.")

            num_firetrucks = count_firetrucks(vl_gpt, vl_chat_processor, tokenizer, image_path)
            if num_firetrucks > 0:
                firetruck_image_ids.append(image_path)
                print(f"  >>>> Firetruck(s) detected: {num_firetrucks}. Added to list.")
            
            num_police_cars = count_police_cars(vl_gpt, vl_chat_processor, tokenizer, image_path)
            if num_police_cars > 0:
                police_car_image_ids.append(image_path)
                print(f"  >>>> Police car(s) detected: {num_police_cars}. Added to list.")

            # Embed the caption and retrieve context for accident detection
            cap_emb = embed_model.encode([caption]).tolist() # embed_model expects a list
            results = collection.query(query_embeddings=cap_emb, n_results=3, include=['documents']) # Only fetch documents
            retrieved = results['documents'][0] if results and results.get('documents') and results['documents'][0] else []
            
            if retrieved:
                print(f"  Retrieved {len(retrieved)} Context Snippets for RAG.")
            else:
                print(f"  No context snippets retrieved for RAG. Caption: {caption}")


            # Perform RAG-based inference to detect accidents
            rag_accident_response = process_with_rag_accident(vl_gpt, vl_chat_processor, tokenizer, image_path, caption, retrieved)
            print(f"  RAG Response (Is Accident?): {rag_accident_response}")

            if rag_accident_response.lower() == 'yes':
                accident_ids.append(image_path)
                print(f"  >>>> Accident detected. Added to list.")

        except FileNotFoundError:
            print(f"  Error: Image file not found at {image_path}. Skipping.")
        except Exception as e:
            print(f"  Error processing image {image_file}: {e}")
            import traceback
            traceback.print_exc()


    print(f"\nFinished processing directory: {photo_dir}")
    print(f"  Accidents detected in this folder: {len(accident_ids)}")
    print(f"  Images with ambulances in this folder: {len(ambulance_image_ids)}")
    print(f"  Images with firetrucks in this folder: {len(firetruck_image_ids)}")
    print(f"  Images with police cars in this folder: {len(police_car_image_ids)}")
    print("-" * 30)

    return accident_ids, ambulance_image_ids, firetruck_image_ids, police_car_image_ids


# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Configuration ---
    # IMPORTANT: Ensure you are logged into Hugging Face if the model requires it.
    # e.g., run `huggingface-cli login` in your terminal, or set HF_TOKEN.
    # setup_hf() can be expanded to handle this if needed.

    base_dir = "C:/Users/admin/Desktop/" 
    input_folders = [
        os.path.join(base_dir, "accident_photos/3"),
        os.path.join(base_dir, "accident_photos/4"),
        os.path.join(base_dir, "accident_photos/5"),
        os.path.join(base_dir, "comparison_photos/2"),
        os.path.join(base_dir, "comparison_photos/6"),
        os.path.join(base_dir, "comparison_photos/7"),
    ]
    chroma_db_path = os.path.join(base_dir, "chroma_db_traffic_v2") # Consider versioning or specific names
    chroma_collection_name = "traffic_incidents_kb"

    # Output file paths
    output_accidents_file = os.path.join(base_dir, "detected_accident_images.txt")
    output_ambulances_file = os.path.join(base_dir, "detected_ambulance_images.txt")
    output_firetrucks_file = os.path.join(base_dir, "detected_firetruck_images.txt")
    output_police_cars_file = os.path.join(base_dir, "detected_police_car_images.txt")

    build_kb_flag = False # Set to True ONLY if you need to (re)build the knowledge base

    # --- 1. Setup ---
    print("Starting setup...")
    setup_hf() 
    vl_gpt, vl_chat_processor, tokenizer = load_vision_model()
    embed_model = load_embed_model()
    collection = setup_chroma(db_path=chroma_db_path, collection_name=chroma_collection_name)
    print("Setup complete.")

    # --- 2. Build Knowledge Base (Conditional) ---
    if build_kb_flag:
        print("\nBuilding Knowledge Base...")
        build_knowledge_base(embed_model, collection)
        print("Knowledge Base build process finished.")
    else:
        print("\nSkipping Knowledge Base build (build_kb_flag is False).")
        if collection.count() == 0:
            print("Warning: Knowledge base collection appears to be empty and build_kb_flag is False. RAG may not be effective.")

    # --- 3. Run Inference Across Folders and Collect Results ---
    print("\nStarting inference process for all configured folders...")
    all_accident_ids = []
    all_ambulance_image_ids = []
    all_firetruck_image_ids = []
    all_police_car_image_ids = []

    for folder_path in input_folders:
        try:
            folder_acc_ids, folder_amb_ids, folder_ft_ids, folder_pc_ids = run_inference(
                vl_gpt, vl_chat_processor, tokenizer, embed_model, collection, folder_path
            )
            all_accident_ids.extend(folder_acc_ids)
            all_ambulance_image_ids.extend(folder_amb_ids)
            all_firetruck_image_ids.extend(folder_ft_ids)
            all_police_car_image_ids.extend(folder_pc_ids)
        except Exception as e:
            print(f"FATAL ERROR during processing of directory {folder_path}: {e}")
            import traceback
            traceback.print_exc()
            print("Attempting to continue with the next directory...")

    # --- 4. Write Accumulated Results to Files ---
    print("\n--- Overall Results ---")
    print(f"Total accident images detected: {len(all_accident_ids)}")
    print(f"Total images with ambulances: {len(all_ambulance_image_ids)}")
    print(f"Total images with firetrucks: {len(all_firetruck_image_ids)}")
    print(f"Total images with police cars: {len(all_police_car_image_ids)}")

    output_files_map = {
        output_accidents_file: ("accident images", all_accident_ids),
        output_ambulances_file: ("images with ambulances", all_ambulance_image_ids),
        output_firetrucks_file: ("images with firetrucks", all_firetruck_image_ids),
        output_police_cars_file: ("images with police cars", all_police_car_image_ids),
    }

    for file_path, (description, id_list) in output_files_map.items():
        print(f"\nWriting {description} to: {file_path}")
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                if not id_list:
                    f.write(f"No {description} detected.\n")
                else:
                    for img_path in id_list:
                        f.write(img_path + "\n")
            print(f"Successfully wrote {len(id_list)} entries to {file_path}.")
        except IOError as e:
            print(f"Error writing results to file {file_path}: {e}")

    print("\nScript finished.")