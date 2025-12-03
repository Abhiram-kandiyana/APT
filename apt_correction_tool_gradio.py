import gradio as gr
import argparse
import json
import os
import sys

# Global state
DATA = []
CURRENT_INDEX = 0
ARGS = None

def parse_args():
    parser = argparse.ArgumentParser(description="APT Caption Correction Tool (Gradio)")
    parser.add_argument("--input_json", required=True, help="Path to input JSON file")
    parser.add_argument("--output_json", required=True, help="Path to output JSON file")
    parser.add_argument("--label_map", nargs='+', required=True, help="List of valid labels")
    return parser.parse_args()

def get_current_data():
    global DATA, CURRENT_INDEX, ARGS
    if not DATA:
        return None, "", "", "No data"
        
    item = DATA[CURRENT_INDEX]
    img_path = item['image_path']
    label = item.get('label', '')
    if label not in ARGS.label_map and ARGS.label_map:
         label = ARGS.label_map[0]
    rationale = item.get('rationale', '')
    
    info_text = f"Image {CURRENT_INDEX + 1} / {len(DATA)}"
    return img_path, label, rationale, info_text

def save_current_state(label, rationale):
    global DATA, CURRENT_INDEX
    if not DATA:
        return
    DATA[CURRENT_INDEX]['label'] = label
    DATA[CURRENT_INDEX]['rationale'] = rationale

def next_image(label, rationale):
    global CURRENT_INDEX, DATA
    save_current_state(label, rationale)
    
    if CURRENT_INDEX < len(DATA) - 1:
        CURRENT_INDEX += 1
        
    return get_current_data()

def prev_image(label, rationale):
    global CURRENT_INDEX, DATA
    save_current_state(label, rationale)
    
    if CURRENT_INDEX > 0:
        CURRENT_INDEX -= 1
        
    return get_current_data()

def save_and_finish(label, rationale):
    global DATA, ARGS
    save_current_state(label, rationale)
    
    with open(ARGS.output_json, 'w') as f:
        json.dump(DATA, f, indent=4)
    
    print(f"Saved corrections to {ARGS.output_json}")
    sys.exit(0)

def main():
    global DATA, ARGS, CURRENT_INDEX
    ARGS = parse_args()
    
    with open(ARGS.input_json, 'r') as f:
        DATA = json.load(f)
    
    if not DATA:
        print("No data to process.")
        return

    with gr.Blocks(title="APT Caption Correction Tool") as demo:
        with gr.Row():
            info_display = gr.Markdown()
            
        with gr.Row():
            with gr.Column(scale=2):
                image_display = gr.Image(type="filepath", label="Image", height=600)
            
            with gr.Column(scale=1):
                label_dropdown = gr.Dropdown(choices=ARGS.label_map, label="Class Label", interactive=True)
                rationale_box = gr.Textbox(label="Rationale", lines=10, interactive=True)
        
        with gr.Row():
            prev_btn = gr.Button("Previous")
            save_btn = gr.Button("Save & Finish", variant="primary")
            next_btn = gr.Button("Next")

        # Initial Load
        initial_img, initial_lbl, initial_rat, initial_info = get_current_data()
        image_display.value = initial_img
        label_dropdown.value = initial_lbl
        rationale_box.value = initial_rat
        info_display.value = initial_info

        # Event Listeners
        prev_btn.click(
            fn=prev_image,
            inputs=[label_dropdown, rationale_box],
            outputs=[image_display, label_dropdown, rationale_box, info_display]
        )
        
        next_btn.click(
            fn=next_image,
            inputs=[label_dropdown, rationale_box],
            outputs=[image_display, label_dropdown, rationale_box, info_display]
        )
        
        save_btn.click(
            fn=save_and_finish,
            inputs=[label_dropdown, rationale_box],
            outputs=[]
        )

    print("Launching Gradio interface... Please check for the public URL below.")
    demo.launch(inbrowser=False, prevent_thread_lock=False, share=True, server_name="0.0.0.0")

if __name__ == "__main__":
    main()
