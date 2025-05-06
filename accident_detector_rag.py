

import os
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images

# --- 1. 登录 Hugging Face ---


# --- 2. 加载 Janus-Pro-7B 视觉模型 ---
def load_vision_model(model_id: str = "deepseek-ai/Janus-Pro-7B"):
    vl_chat_processor = VLChatProcessor.from_pretrained(model_id)
    tokenizer = vl_chat_processor.tokenizer
    
    vl_gpt = AutoModelForCausalLM.from_pretrained(
        model_id, 
        trust_remote_code=True
    )
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()
    
    return vl_gpt, vl_chat_processor, tokenizer

# --- 3. 初始化文本 Embedding 模型（用于 Chroma 检索） ---
def load_embed_model(model_name: str = 'all-MiniLM-L6-v2'):
    return SentenceTransformer(model_name)

# --- 4. 初始化 Chroma 本地向量数据库 ---
def setup_chroma(db_path: str = "c:/users/admin/desktop/chroma_db", collection_name: str = "your_collection_name"):
    # 创建持久化客户端
    client = chromadb.PersistentClient(path=db_path)
    # 创建或获取集合
    collection = client.get_or_create_collection(collection_name)
    return collection

# --- 5. 构建知识库（仅需运行一次，如果已构建可注释掉） ---
def build_knowledge_base(embed_model, collection):
    # List 1: Accident Descriptions
    accident_descriptions = [ 
    "A truck lies overturned on the highway median, surrounded by emergency vehicles and scattered debris.",
    "A red sedan has collided with a guardrail, its hood crumpled and brake lights illuminated.",
    "Two semi-trailers are locked together in a T-bone collision, blocking all lanes of traffic.",
    "A motorcycle rider lies motionless beside their bike after skidding off the wet road.",
    "Flames and smoke rise from a burning vehicle near an exit ramp, with firefighters on scene.",
    "A minivan has veered into a ditch, its wheels spinning futilely in the mud.",
    "Emergency crews attend to a driver trapped inside a crushed pickup truck.",
    "Debris from a shattered windshield litters the road after a high-speed crash.",
    "A school bus has jackknifed across multiple lanes, halting traffic for miles.",
    "Paramedics assess an injured cyclist lying next to a damaged bicycle.",
    "A tractor-trailer’s cargo spills onto the road following a rollover accident.",
    "Brake lights stretch endlessly behind a stalled vehicle in the middle lane.",
    "A car has rear-ended a stopped truck, pushing it into the back of another vehicle.",
    "Smoke billows from a car’s engine compartment after a head-on collision.",
    "An SUV rests on its side in the grassy median, its windows shattered.",
    "Firefighters cut through the roof of a vehicle to rescue a passenger.",
    "A semi-truck’s trailer detaches and slides across the highway during a turn.",
    "Blood stains mark the pavement near a pedestrian struck by a speeding car.",
    "Emergency lights flash around a pileup involving four cars in foggy conditions.",
    "A car’s trunk is embedded in a tree after a violent swerve to avoid an animal.",
    "Rescuers drag a driver from a sinking vehicle in a flooded underpass.",
    "A motorcycle’s front wheel is missing after colliding with a pothole.",
    "Glass shards and tire fragments scatter the road after a chain-reaction crash.",
    "A car’s airbags are deployed mid-air as it flips over a concrete barrier.",
    "Emergency helicopters hover above a remote crash site inaccessible by ground.",
    "A driver exits their smoking vehicle moments before it bursts into flames.",
    "Skid marks lead to a crumpled car wedged against a utility pole.",
    "Medics perform CPR on a driver ejected from a violently spinning vehicle.",
    "A semi-truck’s load of lumber spills across the highway during a sharp turn.",
    "Blood trails lead from a pedestrian lying unconscious in the crosswalk.",
    "A car’s hood is folded inward after slamming into a parked tow truck.",
    "Emergency crews stabilize a patient pinned beneath a collapsed guardrail.",
    "A driver’s side window is shattered by a flying object during a collision.",
    "Flames consume the rear of a van after a fuel tank rupture.",
    "A car’s license plate is bent beyond recognition in a fiery crash.",
    "Rescuers use jaws of life to free a passenger trapped in a mangled wreck.",
    "A semi-truck’s cab is detached from its trailer after a violent rollover.",
    "Blood splatters the windshield of a car involved in a fatal collision.",
    "A motorcycle’s handlebars are twisted into the ground after a high-speed wipeout.",
    "Emergency personnel attend to a driver bleeding from a head injury.",
    "A car’s trunk is crushed by a fallen tree during a storm.",
    "Skid marks form a circular pattern around a spun-out vehicle.",
    "A semi-truck’s cargo of barrels spills flammable liquid onto the road.",
    "Rescuers carry a stretcher past a smoldering wreck on the shoulder.",
    "A driver’s side door is ripped off by a passing vehicle during a sideswipe.",
    "Blood stains the curb where a pedestrian was struck by a fleeing driver.",
    "A car’s roof is peeled back like a sardine can after a rollover.",
    "Emergency crews douse a vehicle engulfed in flames on a rural road.",
    "A semi-truck’s trailer shears off a car’s top during a blind curve.",
    "Medics bandage a driver’s arm severed in a violent crash.",
    "A car’s engine block is torn loose after a collision with a concrete barrier.",
    "Emergency lights reflect off a pool of spilled oil from a wrecked tanker.",
    "A driver’s seatbelt is shredded by a jagged piece of metal from a crash.",
    "Blood drips from the roof of a car where a passenger was impaled.",
    "A semi-truck’s load of livestock scatters across the highway after a crash.",
    "Rescuers lift a child from a submerged vehicle using ropes.",
    "A car’s taillights are melted by the heat of a post-crash fire.",
    "Emergency crews secure a hazardous material spill from a wrecked tanker.",
    "A driver’s hand hangs limply from the steering wheel of a totaled vehicle.",
    "Bloodstains mark the dashboard of a car involved in a fatal impact.",
    "A semi-truck’s cab is split in half after a head-on collision.",
    "Medics apply pressure to a gushing wound on a pedestrian hit by a car.",
    "A car’s doors are blown off by the force of a high-speed crash.",
    "Emergency personnel shield bystanders from a flaming wreck with tarps.",
    "A driver’s side mirror is snapped off during a sideswipe collision.",
    "Blood pools beneath a motorcyclist lying face-down on the asphalt.",
    "A semi-truck’s load of chemicals leaks onto the road after a rollover.",
    "Rescuers extract a passenger from a vehicle submerged in floodwaters.",
    "A car’s tires explode in a fiery crash, sending debris everywhere.",
    "Emergency crews extinguish a vehicle fire caused by a short circuit.",
    "A driver’s side window is shattered by a bullet during a road rage incident.",
    "Blood streaks the side of a car where a passenger was thrown from the vehicle.",
    "A semi-truck’s trailer crushes a car during a violent jackknife.",
    "Medics administer oxygen to a driver suffering smoke inhalation from a crash.",
    "A car’s engine is torn loose after a collision with a train.",
    "Emergency personnel secure a hazardous material spill from a wrecked tanker.",
    "A driver’s side door is caved in by a tree branch during a storm.",
    "Blood stains the center console of a car involved in a violent crash.",
    "A semi-truck’s load of livestock scatters across the highway after a crash.",
    "Rescuers lift a child from a submerged vehicle using ropes.",
    "A car’s taillights are melted by the heat of a post-crash fire.",
    "Emergency crews secure a hazardous material spill from a wrecked tanker.",
    "A driver’s hand hangs limply from the steering wheel of a totaled vehicle.",
    "Bloodstains mark the dashboard of a car involved in a fatal impact.",
    "A semi-truck’s cab is split in half after a head-on collision.",
    "Medics apply pressure to a gushing wound on a pedestrian hit by a car.",
    "A car’s doors are blown off by the force of a high-speed crash.",
    "Emergency personnel shield bystanders from a flaming wreck with tarps.",
    "A driver’s side mirror is snapped off during a sideswipe collision.",
    "Blood pools beneath a motorcyclist lying face-down on the asphalt.",
    "A semi-truck’s load of chemicals leaks onto the road after a rollover.",
    "Rescuers extract a passenger from a vehicle submerged in floodwaters.",
    "A car’s tires explode in a fiery crash, sending debris everywhere.",
    "Emergency crews extinguish a vehicle fire caused by a short circuit.",
    "A driver’s side window is shattered by a bullet during a road rage incident.",
    "Blood streaks the side of a car where a passenger was thrown from the vehicle.",
    "A semi-truck’s trailer crushes a car during a violent jackknife.",
    "Medics administer oxygen to a driver suffering smoke inhalation from a crash.",
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
    "Vehicles cruise smoothly  on a wide, open multi-lane highway.",
    "Cars maintain consistent spacing while travelling quickly along a coastal road.",
    "Traffic flows freely through a green-lit intersection, vehicles accelerating away cleanly.",
    "Headlights and taillights form neat, fast-moving lines on a highway at night.",
    "Vehicles change lanes smoothly and predictably on a freeway with moderate traffic volume.",
    "Cars efficiently merge onto the highway from an on-ramp, matching the speed of flowing traffic.",
    "A steady stream of vehicles moves quickly across a long suspension bridge.",
    "Traffic progresses rapidly through a tunnel, lights reflecting off the pasdsing cars.",
    "Rural highway traffic moves swiftly, with large gaps between vehicles or clusters of cars.",
    "Early morning traffic flows unimpeded before the main rush hour begins.",
    "Vehicles travel at high speed on an autobahn or similar high-speed roadway.",
    "A group of motorcycles cruises comfortably in formation within a lane of fast-moving traffic.",
    "Cars crest a hill on the highway and accelerate smoothly down the other side.",
    "Traffic lights are synchronized, allowing platoons of vehicles to pass through multiple intersections without stopping.",
    "Weekend traffic outside of peak hours flows effortlessly along suburban parkways.",
    "Freshly paved asphalt provides a smooth surface for fast-moving vehicles.",
    "Cars quickly navigate a gentle curve on a well-maintained highway, maintaining speed.",
    "Express lanes show vehicles moving significantly faster than the regular lanes beside them.",
    "Light traffic volume allows vehicles to travel unhindered across the city.",
    "A convertible speeds along a scenic route, top down, wind blowing through the occupants' hair.",
    "Trucks maintain highway speed in the right lanes, while faster cars pass easily on the left.",
    "Traffic flows consistently through an automated toll collection point without slowing down.",
    "Vehicles quickly dissipate after passing through a previously congested area.",
    "Sunlight glints off windshields as cars move rapidly along a straight stretch of road.",
    "Night traffic consists of widely spaced pairs of headlights approaching and taillights receding quickly.",
    "Minimal braking is observed as vehicles smoothly adjust speed to maintain flow.",
    "Cars exit the highway onto off-ramps without causing backups.",
    "Traffic moves briskly over rolling hills on a country highway.",
    "The sound is primarily tire noise on pavement rather than engines laboring or horns honking.",
    "Newly opened highway section shows light, fast traffic enjoying the clear road.",
    "Vehicles easily overtake slower traffic, utilizing available passing lanes.",
    "Consistent flow is maintained even during moderate rain, indicating good driving conditions and behavior.",
    "Traffic speeds through a recently cleared construction zone, returning to normal pace.",
    "A police car cruises at speed with traffic, not impeding the flow.",
    "Green lights stretch ahead down a long city avenue, allowing uninterrupted travel for blocks.",
    "Cars accelerate quickly away from a toll booth after paying.",
    "High-occupancy vehicle (HOV) lanes show rapid movement compared to potentially slower adjacent lanes.",
    "Digital signs overhead display the speed limit, which matches the average speed of traffic.",
    "Birds-eye view shows vehicles moving like clockwork through a complex but efficient interchange.",
    "Passing lanes on uphill sections allow faster vehicles to maintain momentum.",
    "Traffic flows smoothly around a wide, sweeping curve on a bypass road.",
    "The road surface is clear of debris, snow, or water, facilitating high-speed travel.",
    "Ample space between vehicles allows for safe following distances even at higher speeds.",
    "Drivers appear relaxed and focused, navigating the fluent traffic conditions easily.",
    "Late-night freeway traffic moves very quickly due to low volume.",
    "A stream of cars efficiently navigates a series of S-curves on a well-engineered road.",
    "Sunlight reflects brightly off the clean surfaces of fast-moving cars.",
    "Traffic flow remains stable and fast despite merging lanes up ahead.",
    "Vehicles crossing a border checkpoint move swiftly after passing through inspection.",
    "A long stretch of desert highway sees vehicles travelling at high, consistent speeds with vast distances between them.",
        "Cars zip past with barely a pause on the open highway.",
    "Vehicles cruise steadily under clear blue skies.",
    "Traffic flows smoothly with no signs of congestion ahead.",
    "Trucks keep to their lanes at full highway speed.",
    "Motorcyclists weave effortlessly between slow-moving cars.",
    "Cars merge fluidly onto the expressway with no delays.",
    "Drivers accelerate smoothly after passing through a green light.",
    "The road ahead is clear, with cars spaced perfectly apart.",
    "A convoy of vehicles moves uniformly down a wide avenue.",
    "No traffic lights or stops interrupt a peaceful cruise.",
    "Trucks maintain a steady pace on the interstate.",
    "Sports cars overtake slower vehicles with ease.",
    "Traffic flows in rhythm through synchronized lights downtown.",
    "The early morning commute is fast and hassle-free.",
    "Winds whip past as vehicles fly down a mountain road.",
    "Long stretches of road allow uninterrupted driving.",
    "Cars move quickly through well-marked highway lanes.",
    "A downhill stretch boosts traffic speed naturally.",
    "Drivers pass through toll booths with high-speed tags.",
    "Light traffic allows constant high speeds through town.",
    "Vehicles maintain cruise control settings without braking.",
    "Drivers keep to the limit with no obstacles in sight.",
    "All lanes are clear, encouraging consistent flow.",
    "Highway patrol watches calmly as traffic moves efficiently.",
    "An overpass eliminates congestion from intersections below.",
    "Motorists navigate turns without slowing significantly.",
    "Exit ramps are clear, enabling fast merging and exiting.",
    "Drivers feel the breeze through rolled-down windows on the move.",
    "The road hums beneath tires at a brisk pace.",
    "Lane markings guide cars effortlessly through smooth curves.",
    "There are no traffic alerts or disruptions for miles.",
    "The speed limit signs are respected but rarely tested.",
    "Even large trucks glide along without needing to brake.",
    "Green arrows light up just in time at every intersection.",
    "Fast lanes stay fast as slower traffic keeps right.",
    "The open countryside highway stretches without a car in sight.",
    "Sunlight reflects off speeding car roofs on an empty road.",
    "Wide shoulders and good signage aid high-speed travel.",
    "Road conditions are ideal for fast movement.",
    "Navigation apps report 'no delays' in all directions.",
    "Cars coast smoothly under overhead highway lighting.",
    "Motorcycles pass in streaks of light on a clear freeway.",
    "Light winds and dry roads aid swift driving.",
    "The four-lane bypass keeps traffic light and fast.",
    "Urban highways are unusually empty this morning.",
    "Vehicles make efficient use of green-light corridors.",
    "Cars accelerate onto freeways without checking mirrors.",
    "Drivers maintain high speeds through sharp highway curves.",
    "The night drive is seamless with zero slowdowns.",
    "Fast traffic flows seamlessly under a cloudless sky."
]


    # Combine
    all_texts = accident_descriptions + heavy_traffic_descriptions + fast_traffic_descriptions
    # Generate unique IDs
    all_ids = [f"desc_{i}" for i in range(len(all_texts))]
    # Generate metadata
    all_metadatas = (
        [{"is_accident": True} for _ in accident_descriptions] +
        [{"is_accident": False} for _ in heavy_traffic_descriptions] +
        [{"is_accident": False} for _ in fast_traffic_descriptions]
    )

    # Embedding & upsert
    embeddings = embed_model.encode(all_texts).tolist()
    collection.upsert(
        ids=all_ids,
        embeddings=embeddings,
        documents=all_texts,
        metadatas=all_metadatas
    )
    print("Knowledge base built with", len(all_texts), "entries.")

# --- 6. 使用 Janus 模型生成图像描述 ---
def generate_image_description(vl_gpt, vl_chat_processor, tokenizer, image_path):
    # 构建对话
    conversation = [
        {
            "role": "<|User|>",
            "content": "<image_placeholder>\nDescribe the image in one sentence.",
            "images": [image_path],  # Pass the image path instead of PIL Image object
        },
        {"role": "<|Assistant|>", "content": ""},
    ]
    
    # 加载图像并准备输入
    pil_images = load_pil_images(conversation)
    prepare_inputs = vl_chat_processor(
        conversations=conversation, images=pil_images, force_batchify=True
    ).to(vl_gpt.device)
    
    # 运行图像编码器获取图像嵌入
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
    
    # 运行模型获取响应
    outputs = vl_gpt.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=100,
        do_sample=False,
        use_cache=True,
    )
    
    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    return answer

# --- 7. 使用 Janus 模型进行 RAG 推理 ---
def process_with_rag(vl_gpt, vl_chat_processor, tokenizer, image_path, caption, retrieved_context):
    # 构建 RAG 提示词
    rag_prompt = f"Image description: {caption}\n"
    rag_prompt += "Retrieved knowledge:\n"
    for i, doc in enumerate(retrieved_context, start=1):
        rag_prompt += f"{i}. {doc}\n"
    rag_prompt += "Question: Are there any accident? Respond solely with 'yes' or 'no'."
    
    # 构建对话
    conversation = [
        {
            "role": "<|User|>",
            "content": f"<image_placeholder>\n{rag_prompt}",
            "images": [image_path],  # Pass the image path instead of PIL Image object
        },
        {"role": "<|Assistant|>", "content": ""},
    ]
    
    # 加载图像并准备输入
    pil_images = load_pil_images(conversation)
    prepare_inputs = vl_chat_processor(
        conversations=conversation, images=pil_images, force_batchify=True
    ).to(vl_gpt.device)
    
    # 运行图像编码器获取图像嵌入
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
    
    # 运行模型获取响应
    outputs = vl_gpt.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=5,
        do_sample=False,
        use_cache=True,
    )
    
    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    # 提取答案中的"yes"或"no"
    answer = answer.lower().strip()
    if "yes" in answer:
        return "yes"
    elif "no" in answer:
        return "no"
    else:
        return answer

# --- 8. 图像批量处理 + RAG 推理 ---
def run_inference(vl_gpt, vl_chat_processor, tokenizer, embed_model, collection, photo_dir):
    """
    Processes all images in a given directory to detect accidents using RAG.

    Args:
        vl_gpt: Loaded vision language model.
        vl_chat_processor: Processor for the vision language model.
        tokenizer: Tokenizer for the language model.
        embed_model: Sentence embedding model.
        collection: ChromaDB collection object.
        photo_dir: Path to the directory containing images.

    Returns:
        A list of image IDs (filenames without extension) identified as accidents
        in this directory.
    """
    accident_ids_in_dir = [] # Local list for this directory's results
    print("-" * 30)
    print(f"Processing directory: {photo_dir}")

    if not os.path.isdir(photo_dir):
        print(f"Error: Directory not found: {photo_dir}")
        return [] # Return empty list if directory doesn't exist

    # Get list of image files
    try:
        image_files = [f for f in os.listdir(photo_dir)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]
    except OSError as e:
        print(f"Error accessing directory {photo_dir}: {e}")
        return []

    if not image_files:
        print("No image files found in this directory.")
        return []

    print(f"Found {len(image_files)} images to process.")

    for image_file in image_files:
        image_path = os.path.join(photo_dir, image_file)
        image_id = os.path.splitext(image_file)[0] # Filename without extension

        print(f"\nProcessing image: {image_file} (ID: {image_id})")

        try:
            # 8.1 使用 Janus 生成图像描述
            caption = generate_image_description(vl_gpt, vl_chat_processor, tokenizer, image_path)
            print(f"  Generated Caption: {caption}")

            # 8.2 Embedding & 检索
            cap_emb = embed_model.encode([caption]).tolist()
            results = collection.query(query_embeddings=cap_emb, n_results=3) # Retrieve top 3 most similar docs
            retrieved = results['documents'][0] if results and results['documents'] else []
            print(f"  Retrieved Context Snippets: {len(retrieved)}")
            # for i, doc in enumerate(retrieved): # Optional: print retrieved docs
            #     print(f"    {i+1}. {doc}")

            # 8.3 使用 Janus 和 RAG 进行推理
            resp = process_with_rag(vl_gpt, vl_chat_processor, tokenizer, image_path, caption, retrieved)
            print(f"  RAG Response (Is Accident?): {resp}")

            # 8.4 Record if it's an accident
            if resp == 'yes':
                accident_ids_in_dir.append(image_id)
                print(f"  Result: Accident detected for {image_id}")

        except FileNotFoundError:
            print(f"  Error: Image file not found at {image_path}. Skipping.")
        except Exception as e:
            # Catch other potential errors during processing (model errors, etc.)
            print(f"  Error processing image {image_file}: {e}")
            # Continue to the next image

    print(f"Finished processing directory: {photo_dir}")
    print(f"Accidents detected in this directory: {len(accident_ids_in_dir)}")
    print("-" * 30)
    return accident_ids_in_dir # Return the list of IDs for this directory

# --- Main Execution Block (Rewritten) ---
if __name__ == "__main__":
    # --- Configuration ---
    # Use environment variable for token (replace placeholder in setup_hf)
    # os.environ['HF_TOKEN'] = 'YOUR_HF_TOKEN_HERE' # Set it before running script

    # Define input image folders and output file path
    # Make these paths absolute or ensure they are relative to the script's execution location
    base_dir = "C:/Users/admin/Desktop/" # Define base path for easier modification
    input_folders = [
        os.path.join(base_dir, "accident_photos/3"),
        os.path.join(base_dir, "accident_photos/4"),
        os.path.join(base_dir, "accident_photos/5"),
        os.path.join(base_dir, "comparison_photos/2"),
        os.path.join(base_dir, "comparison_photos/6"),
        os.path.join(base_dir, "comparison_photos/7"),
    ]
    # Consider making Chroma path configurable too
    chroma_db_path = os.path.join(base_dir, "chroma_db_traffic")
    output_file_path = os.path.join(base_dir, "detected_accidents.txt")
    build_kb_flag = False # Set to True if you need to build/rebuild the knowledge base

    # --- 1. Setup ---
    print("Starting setup...")
    setup_hf() # Ensure HF Login works or handles failure
    vl_gpt, vl_chat_processor, tokenizer = load_vision_model()
    embed_model = load_embed_model()
    collection = setup_chroma(db_path=chroma_db_path) # Use configured path
    print("Setup complete.")

    # --- 2. Build Knowledge Base (Conditional) ---
    if build_kb_flag:
        print("\nBuilding Knowledge Base...")
        # Check if collection exists and maybe prompt user if overwriting?
        # For simplicity, just build/rebuild if flag is True
        build_knowledge_base(embed_model, collection)
        print("Knowledge Base build process finished.")
    else:
        print("\nSkipping Knowledge Base build (build_kb_flag is False).")
        # Optional: Check if collection is empty and warn if it is
        if collection.count() == 0:
             print("Warning: Knowledge base collection appears to be empty.")


    # --- 3. Run Inference Across Folders and Collect Results ---
    print("\nStarting inference process...")
    all_accident_ids = [] # Initialize list to store all results

    for folder_path in input_folders:
        # Call run_inference for the current folder
        # Add error handling in case run_inference itself fails unexpectedly
        try:
            folder_accident_ids = run_inference(
                vl_gpt, vl_chat_processor, tokenizer, embed_model, collection, folder_path
            )
            # Extend the main list with results from this folder
            all_accident_ids.extend(folder_accident_ids)
        except Exception as e:
            print(f"FATAL ERROR during processing of directory {folder_path}: {e}")
            print("Attempting to continue with the next directory...")


    # --- 4. Write Accumulated Results to File ---
    print("\nInference complete for all directories.")
    print(f"Total accident images detected across all folders: {len(all_accident_ids)}")
    print(f"List of detected accident image IDs: {all_accident_ids}")

    print(f"\nWriting results to: {output_file_path}")
    try:
        with open(output_file_path, "w", encoding="utf-8") as f:
            if not all_accident_ids:
                f.write("No accidents detected.\n")
            else:
                for img_id in all_accident_ids:
                    f.write(img_id + "\n") # Write each ID on a new line
        print("Successfully wrote results to file.")
    except IOError as e:
        print(f"Error writing results to file {output_file_path}: {e}")

    print("\nScript finished.")