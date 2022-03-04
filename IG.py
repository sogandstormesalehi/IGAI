# In[33]:


import matplotlib.pylab as matPlot
import tensorflow as tf
import tensorflow_hub as hub
import numpy as numpy


# In[34]:


model = tf.keras.Sequential([hub.KerasLayer(name = 'inception_v1', handle = 'https://tfhub.dev/google/imagenet/inception_v1/classification/5', trainable = False),])
model.build([None, 224, 224, 3])
model.summary()


# In[35]:


def load_imagenet_labels(file_path):
    labels_file = tf.keras.utils.get_file('ImageNetLabels.txt', file_path)
    with open(labels_file) as reader:
        file_to_read = reader.read()
        labels = file_to_read.splitlines()
    return numpy.array(labels)    


# In[36]:


imagenet_labels = load_imagenet_labels('https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')


# In[5]:


def read_image(file_name):
    image = tf.io.read_file(file_name)
    image = tf.io.decode_jpeg(image, channels = 3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize_with_pad(image, target_height = 224, target_width = 224)
    return image


# In[6]:


img_url = {
    'First': 'https://i.pinimg.com/564x/75/d2/20/75d22063480fc9f500c029ce68d52d47.jpg',
    'Second': 'https://i.pinimg.com/564x/c7/4a/3c/c74a3cf402a9610fa0a5c5680d297cc1.jpg',
}
img_paths = {name: tf.keras.utils.get_file(name, url) for (name, url) in img_url.items()}
img_name_tensors = {name: read_image(img_path) for (name, img_path) in img_paths.items()}


# In[7]:


matPlot.figure(figsize = (8, 8))
for n, (name, img_tensors) in enumerate(img_name_tensors.items()):
    ax = matPlot.subplot(1, 2, n + 1)
    ax.imshow(img_tensors)
    ax.set_title(name)
    ax.axis('off')
matPlot.tight_layout()


# In[8]:


def top_k_predictedictions(img, k = 3):
    image_batch = tf.expand_dims(img, 0)
    predictedictions = model(image_batch)
    probs = tf.nn.softmax(predictedictions, axis = -1)
    top_probs, top_idxs = tf.math.top_k(inumpyut = probs, k = k)
    top_labels = imagenet_labels[tuple(top_idxs)]
    return top_labels, top_probs[0]


# In[9]:


for (name, img_tensor) in img_name_tensors.items():
    matPlot.imshow(img_tensor)
    matPlot.title(name, fontweight='bold')
    matPlot.axis('off')
    matPlot.show()
    
    predicted_label, predicted_prob = top_k_predictedictions(img_tensor)
    for label, prob in zip(predicted_label, predicted_prob):
        print(f'{label}: {prob:0.1%}')


# In[56]:


baseline = tf.random.uniform(shape=(224,224,3), minval=0.0, maxval=1.0)
matPlot.imshow(baseline)
matPlot.title("Baseline")
matPlot.axis('off')
matPlot.show()


# In[11]:


steps_to_pass = 44
alphas = tf.linspace(start = 0.0, stop = 1.0, num = steps_to_pass + 1)


# In[12]:


def interpolate_images(baseline, image, alphas):
    alphas_x = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]
    baseline_x = tf.expand_dims(baseline, axis = 0)
    inumpyut_x = tf.expand_dims(image, axis = 0)
    delta = inumpyut_x - baseline_x
    images = baseline_x + alphas_x * delta
    return images


# In[13]:


interpolated_images = interpolate_images(
    baseline=baseline,
    image=img_name_tensors['First'],
    alphas=alphas)


# In[14]:


fig = matPlot.figure(figsize = (20, 20))
i = 0
for alpha, image in zip(alphas[0::10], interpolated_images[0::10]):
    i += 1
    matPlot.subplot(1, len(alphas[0::10]), i)
    matPlot.title(f'alpha: {alpha:.1f}')
    matPlot.imshow(image)
    matPlot.axis('off')

matPlot.tight_layout()    


# In[15]:


def compute_gradients(images, target_class_idx):
    with tf.GradientTape() as tape:
        tape.watch(images)
        logits = model(images)
        probs = tf.nn.softmax(logits, axis = -1)[:, target_class_idx]
    return tape.gradient(probs, images)    


# In[16]:


path_gradients = compute_gradients(
images = interpolated_images, target_class_idx = 555)


# In[17]:


print(path_gradients.shape)


# In[18]:


predicted = model(interpolated_images)
predicted_probability = tf.nn.softmax(predicted, axis = -1)[:, 555]


# In[48]:


def integral_approximation(gradients):
    integrated_gradients = tf.math.reduce_mean(gradients / 2, axis = 0)
    return integrated_gradients


# In[20]:


def integrated_gradients(baseline,
                         image,
                         target_class_idx,
                         steps_to_pass=44,
                         batch_size=32):
    alphas = tf.linspace(start = 0.0, stop = 1.0, num = steps_to_pass + 1)
    gradient_batches = []
    for alpha in tf.range(0, len(alphas), batch_size):
        from_ = alpha
        to = tf.minimum(from_ + batch_size, len(alphas))
        alpha_batch = alphas[from_ : to]
        gradient_batch = one_batch(baseline, image, alpha_batch, target_class_idx)
        gradient_batches.append(gradient_batch)
    
    total_gradients = tf.concat(gradient_batches, axis = 0)
    average_gradients = integral_approximation(gradients = total_gradients)
    integrated_gradients = (image - baseline) * average_gradients
    return integrated_gradients


# In[21]:


@tf.function
def one_batch(baseline, image, alpha_batch, target_class_idx):
    interpolated_path_inumpyut_batch = interpolate_images(baseline=baseline,
                                                       image=image,
                                                       alphas=alpha_batch)
    gradient_batch = compute_gradients(images=interpolated_path_inumpyut_batch,
                                       target_class_idx = target_class_idx)
    return gradient_batch


# In[41]:


def plot_img_attributions(baseline,
                          image,
                          target_class_idx,
                          steps_to_pass=tf.constant(50),
                          cmap=None,
                          overlay_alpha=0.4):
    attributions = integrated_gradients(baseline=baseline,
                                      image=image,
                                      target_class_idx=target_class_idx,
                                      steps_to_pass=steps_to_pass)
    attribution_mask = tf.reduce_sum(tf.math.abs(attributions), axis=-1)

    fig, axs = matPlot.subplots(nrows=2, ncols=2, squeeze=False, figsize=(8, 8))
    axs[0, 0].set_title('Baseline Image')
    axs[0, 0].imshow(baseline)
    axs[0, 0].axis('off')

    axs[0, 1].set_title('Original Image')
    axs[0, 1].imshow(image)
    axs[0, 1].axis('off')
    
    axs[1, 0].set_title('Attribution Mask')
    axs[1, 0].imshow(attribution_mask, cmap=cmap)
    axs[1, 0].axis('off')

    axs[1, 1].set_title('Overlay')
    axs[1, 1].imshow(attribution_mask, cmap=cmap)
    axs[1, 1].imshow(image, alpha=overlay_alpha)
    axs[1, 1].axis('off')
    matPlot.tight_layout()
    return fig


# In[49]:


_ = plot_img_attributions(image=img_name_tensors['First'],
                          baseline=baseline,
                          target_class_idx=555,
                          steps_to_pass=240,
                          cmap=matPlot.cm.inferno,
                          overlay_alpha=0.4)


# In[50]:


_ = plot_img_attributions(image=img_name_tensors['Second'],
                          baseline=baseline,
                          target_class_idx=555,
                          steps_to_pass=240,
                          cmap=matPlot.cm.inferno,
                          overlay_alpha=0.4)


# In[ ]:




