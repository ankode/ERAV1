import lightning.pytorch as pl
import torch.nn as nn
import gradio as gr
from torchvision import transforms
import itertools
from models_lt import *
from utils import *
import os
import random
from PIL import Image
import numpy as np

def read_images(directory, n):
    files = os.listdir(directory)
    random_files = random.sample(files, n)
    image_list = []
    for file in random_files:
        if file.endswith('.jpg') or file.endswith('.png'):  # add more conditions if there are other image types
            image = Image.open(os.path.join(directory, file))
            image_array = np.array(image)
            image_list.append(image_array)
    return image_list



transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


def inference(image,gradcam,num_gradcam,opacity,layer,misclassified,num_misclassified,topk):
    model =  CustomResnet()
    model.load_state_dict(torch.load("cifar10_model.pth",map_location=torch.device('cpu')), strict=False)
    softmax = nn.Softmax(dim=1)
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    input = transform(image)
    input = input.unsqueeze(0)
    output = model(input)
    probs = softmax(output).flatten()
    confidences = {class_names[i]: float(probs[i]) for i in range(10)}
    sorted_confidences = dict(sorted(confidences.items(), key=lambda item: item[1],reverse = True))
    confidence_score = dict(itertools.islice(sorted_confidences.items(), topk))
    pred = probs.argmax(dim=0, keepdim=True)
    pred = pred.item()
    if gradcam == 'Yes':
      target_layers = [model.res_block3[3*layer]]
      cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
      image = input.cpu().numpy()
      grayscale_cam = cam(input_tensor=input, targets=[ClassifierOutputTarget(pred)],aug_smooth=True,eigen_smooth=True)
      grayscale_cam = grayscale_cam[0, :]
      visualization = show_cam_on_image(imshow(image.squeeze(0)), grayscale_cam, use_rgb=True,image_weight=opacity)
      print(imshow(image.squeeze(0)).shape)
      if misclassified == 'Yes':
        misclass = read_images('outputs/misclass', n=num_misclassified)
        cam = read_images('outputs/cam', n=num_gradcam)
        return confidence_score,visualization,misclass,cam
      else:
        cam = read_images('outputs/cam', n=num_gradcam)
        return confidence_score,visualization,None,cam
    else:
      if misclassified == 'Yes':
        misclass = read_images('outputs/misclass', n=num_misclassified)
        return confidence_score,None,misclass,None
      else:
        return confidence_score,None,None,None

markdown_content = """
<h1 style="text-align: center;">CustomResNet Classifier Demo - Cifar10</h1>

<p>This is a demonstration of a custom ResNet model trained on the Cifar10 dataset. Test the classifier, explore GradCAM outputs, and view misclassified images.</p>

<h3>Features:</h3>
<ul>
    <li>Test the custom ResNet model on your own images.</li>
    <li>Visualize GradCAM outputs to understand which parts of an image influenced the predictions.</li>
    <li>View misclassified images to gain insights into the model's strengths and weaknesses.</li>
</ul>
"""
with gr.Blocks() as demo:
  gr.Markdown(markdown_content)
  with gr.Row() as interface:
    with gr.Column() as input_panel:
      image = gr.Image(shape=(32,32))

      gradcam = gr.Radio(label="Do you want to see GradCAM output?", choices=["Yes", "No"], default="No")

      with gr.Column(visible=False) as gradcam_details:
        num_gradcam = gr.Slider(minimum=0, maximum=10, value=0, step=1, label="Number of GradCAM Images",
                                description="Select the number of images to visualize with GradCAM overlay.")
        
        opacity = gr.Slider(minimum=0, maximum=1, value=0.5, step=0.1, label="Opacity of Overlay",
                            description="Adjust the opacity of the GradCAM overlay on the image.")
    
        layer = gr.Slider(minimum=-2, maximum=-1, value=-1, step=1, label="Layer for Visualization",
                          description="Choose the model layer for which GradCAM visualization will be generated.")
      
      def filter_gradcam(gradcam):
        if gradcam == 'Yes':
          return gr.update(visible=True)
        else:
          return gr.update(visible=False)
        
      gradcam.change(filter_gradcam, gradcam, gradcam_details)

      misclassified = gr.Radio(label="Do you want to see misclassified Images", choices=["Yes", "No"])

      with gr.Column(visible=False) as misclassified_details:
          num_misclassified = gr.Slider(minimum = 0, maximum=10, value = 0,step=1, label="Number of Misclassified Images")

      def filter_misclassified(misclassified):
        if misclassified == 'Yes':
          return gr.update(visible=True)
        else:
          return gr.update(visible=False)
        
      misclassified.change(filter_misclassified, misclassified, misclassified_details)

      topk = gr.Slider(minimum = 1, maximum=10, value = 1, step=1, label="Number of Classes")
      btn = gr.Button("Classify")

    with gr.Column() as output_panel:
      gradcam_output = gr.Image(shape=(32, 32), label="Output").style(height=240, width=240)
      output_labels = gr.Label(num_top_classes=10)
      misclassified_gallery = gr.Gallery(label="Misclassified Images")
      gradcam_gallery = gr.Gallery(label="Some More GradCam Outputs")


  
  btn.click(fn=inference, inputs=[image,gradcam,num_gradcam,opacity,layer,misclassified,num_misclassified,topk], outputs=[output_labels,gradcam_output,misclassified_gallery,gradcam_gallery])

  gr.Examples(
        [
            ['examples/cat1.jpg', 'Yes', 10, 0.5, -1, 'Yes', 10, 3],
            ['examples/deer1.jpg', 'Yes', 11, 0.5, -1, 'Yes', 7, 4],
            ['examples/bird1.jpg', 'Yes', 14, 0.5, -2, 'Yes', 17, 5],
            ['examples/automobile2.jpg', 'Yes', 2, 0.5, -2, 'Yes', 15, 10],
            ['examples/deer2.jpg', 'Yes', 11, 0.5, -1, 'Yes', 18, 4],
            ['examples/cat2.jpg', 'Yes', 10, 0.5, -2, 'No', 1, 10],
            ['examples/automobile2.jpg', 'Yes', 2, 0.5, -1, 'No', 15, 6],
            ['examples/bird1.jpg', 'No', 15, 0.5, -2, 'Yes', 4, 10],
            ['examples/truck2.jpg', 'Yes', 15, 0.5, -2, 'Yes', 6, 9],
            ['examples/dog1.jpg', 'No', 1, 0.5, -2, 'No', 18, 8]
        ],
        [image,gradcam,num_gradcam,opacity,layer,misclassified,num_misclassified,topk],
        [output_labels,gradcam_output,misclassified_gallery,gradcam_gallery],
        inference,
        cache_examples=True,
    )

if __name__ == "__main__":
    demo.launch()