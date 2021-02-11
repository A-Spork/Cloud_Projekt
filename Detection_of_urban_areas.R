library(sp)
library(keras)
library(tensorflow)
library(tfdatasets)
library(purrr)
library(ggplot2)
library(rsample)
library(stars)
library(raster)
library(reticulate)
library(mapview)
library(imager)
library(raster)
library(gdalUtils)
library(stars)

setwd("./")


# ################image based#############

first_model <- keras_model_sequential()
layer_conv_2d(first_model,filters = 32,kernel_size = 3, activation = "relu",input_shape = c(128,128,3))


layer_max_pooling_2d(first_model, pool_size = c(2, 2)) 
layer_conv_2d(first_model, filters = 64, kernel_size = c(3, 3), activation = "relu") 
layer_max_pooling_2d(first_model, pool_size = c(2, 2)) 
layer_conv_2d(first_model, filters = 128, kernel_size = c(3, 3), activation = "relu") 
layer_max_pooling_2d(first_model, pool_size = c(2, 2)) 
layer_conv_2d(first_model, filters = 128, kernel_size = c(3, 3), activation = "relu")
layer_max_pooling_2d(first_model, pool_size = c(2, 2)) 
layer_flatten(first_model) 
layer_dense(first_model, units = 256, activation = "relu")
layer_dense(first_model, units = 1, activation = "sigmoid")


# Einladen der Trainingsdaten

subset_list <- list.files("Data/Subsets_128/Slices_Muenster/True", full.names = T)
data_true <- data.frame(img=subset_list,lbl=rep(1L,length(subset_list)))
subset_list <- list.files("Data/Subsets_128/Slices_Muenster/False", full.names = T)
data_false <- data.frame(img=subset_list,lbl=rep(0L,length(subset_list)))
data <- rbind(data_true,data_false)
set.seed(2020)

#Verarbeitung der Trainingsdaten
data <- initial_split(data,prop = 0.75, strata = "lbl")


c(nrow(training(data)[training(data)$lbl==0,]), nrow(training(data)[training(data)$lbl==1,]))


training_dataset <- tensor_slices_dataset(training(data))

dataset_iterator <- as_iterator(training_dataset)
dataset_list <- iterate(dataset_iterator)
head(dataset_list)


subset_size <- first_model$input_shape[2:3]
training_dataset <- 
  dataset_map(training_dataset, function(.x)
    list_modify(.x, img = tf$image$decode_jpeg(tf$io$read_file(.x$img))))

training_dataset <- 
  dataset_map(training_dataset, function(.x)
    list_modify(.x, img = tf$image$convert_image_dtype(.x$img, dtype = tf$float32)))

training_dataset <- 
  dataset_map(training_dataset, function(.x)
    list_modify(.x, img = tf$image$resize(.x$img, size = shape(subset_size[1], subset_size[2]))))



training_dataset <- dataset_shuffle(training_dataset, buffer_size = 10L*128)
training_dataset <- dataset_batch(training_dataset, 10L)
training_dataset <- dataset_map(training_dataset, unname)



dataset_iterator <- as_iterator(training_dataset)
dataset_list <- iterate(dataset_iterator)
dataset_list[[1]][[1]]



dataset_list[[1]][[1]]$shape
dataset_list[[1]][[2]]

validation_dataset <- tensor_slices_dataset(testing(data))

validation_dataset <- 
  dataset_map(validation_dataset, function(.x)
    list_modify(.x, img = tf$image$decode_jpeg(tf$io$read_file(.x$img))))

validation_dataset <- 
  dataset_map(validation_dataset, function(.x)
    list_modify(.x, img = tf$image$convert_image_dtype(.x$img, dtype = tf$float32)))

validation_dataset <- 
  dataset_map(validation_dataset, function(.x)
    list_modify(.x, img = tf$image$resize(.x$img, size = shape(subset_size[1], subset_size[2]))))

validation_dataset <- dataset_batch(validation_dataset, 10L)
validation_dataset <- dataset_map(validation_dataset, unname)


# Trainieren des Modells

compile(
  first_model,
  optimizer = optimizer_rmsprop(lr = 5e-5),
  loss = "binary_crossentropy",
  metrics = "accuracy"
)



diagnostics <- fit(first_model,
                   training_dataset,
                   epochs =  18,
                   validation_data = validation_dataset)

plot(diagnostics)


# Vorhersage
predictions <- predict(first_model,validation_dataset)


par(mfrow=c(1,3),mai=c(0.1,0.1,0.3,0.1),cex=0.8)
for(i in 1:3){
  sample <- floor(runif(n = 1,min = 1,max = 56))
  img_path <- as.character(testing(data)[[sample,1]])
  img <- stack(img_path)
  plotRGB(img,margins=T,main = paste("prediction:",round(predictions[sample],digits=3)," | ","label:",as.character(testing(data)[[sample,2]])))
}

# Einladen der Validierungsdaten
subset_list <- list.files("Data/Subsets_128/Slices_Berlin/True", full.names = T)

dataset <- tensor_slices_dataset(subset_list)
dataset <- dataset_map(dataset, function(.x) 
  tf$image$decode_jpeg(tf$io$read_file(.x))) 
dataset <- dataset_map(dataset, function(.x) 
  tf$image$convert_image_dtype(.x, dtype = tf$float32)) 
dataset <- dataset_map(dataset, function(.x) 
  tf$image$resize(.x, size = shape(128, 128))) 
dataset <- dataset_batch(dataset, 10L)
dataset <-  dataset_map(dataset, unname)

#Vorhersage auf den Validierungsdaten
predictions <- predict(first_model, dataset)

save_model_hdf5(first_model,filepath = "Data/Models/imagebased_model2.h5")



# Validierung
vgg16_feat_extr <- application_vgg16(include_top = F,input_shape = c(128,128,3),weights = "imagenet")
freeze_weights(vgg16_feat_extr)
pretrained_model <- keras_model_sequential(vgg16_feat_extr$layers[1:15])

pretrained_model <- layer_flatten(pretrained_model)
pretrained_model <- layer_dense(pretrained_model,units = 256,activation = "relu")
pretrained_model <- layer_dense(pretrained_model,units = 1,activation = "sigmoid")


compile(
  pretrained_model,
  optimizer = optimizer_rmsprop(lr = 1e-5),
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

diagnostics <- fit(pretrained_model,
                   training_dataset,
                   epochs = 8,
                   validation_data = validation_dataset)
plot(diagnostics)

diagnostics$metrics










#UNET########################################################

input_tensor <- layer_input(shape = c(448,448,3))

unet_tensor <- layer_conv_2d(input_tensor,filters = 64,kernel_size = c(3,3), padding = "same",activation = "relu")
conc_tensor2 <- layer_conv_2d(unet_tensor,filters = 64,kernel_size = c(3,3), padding = "same",activation = "relu")
unet_tensor <- layer_max_pooling_2d(conc_tensor2)

unet_tensor <- layer_conv_2d(unet_tensor,filters = 128,kernel_size = c(3,3), padding = "same",activation = "relu")
conc_tensor1 <- layer_conv_2d(unet_tensor,filters = 128,kernel_size = c(3,3), padding = "same",activation = "relu")
unet_tensor <- layer_max_pooling_2d(conc_tensor1)

unet_tensor <- layer_conv_2d(unet_tensor,filters = 256,kernel_size = c(3,3), padding = "same",activation = "relu")
unet_tensor <- layer_conv_2d(unet_tensor,filters = 256,kernel_size = c(3,3), padding = "same",activation = "relu")

unet_tensor <- layer_conv_2d_transpose(unet_tensor,filters = 128,kernel_size = c(2,2),strides = 2,padding = "same") 
unet_tensor <- layer_concatenate(list(conc_tensor1,unet_tensor))
unet_tensor <- layer_conv_2d(unet_tensor, filters = 128, kernel_size = c(3,3),padding = "same", activation = "relu")
unet_tensor <- layer_conv_2d(unet_tensor, filters = 128, kernel_size = c(3,3),padding = "same", activation = "relu")

unet_tensor <- layer_conv_2d_transpose(unet_tensor,filters = 64,kernel_size = c(2,2),strides = 2,padding = "same")
unet_tensor <- layer_concatenate(list(conc_tensor2,unet_tensor))
unet_tensor <- layer_conv_2d(unet_tensor, filters = 64, kernel_size = c(3,3),padding = "same", activation = "relu")
unet_tensor <- layer_conv_2d(unet_tensor, filters = 64, kernel_size = c(3,3),padding = "same", activation = "relu")

unet_tensor <- layer_conv_2d(unet_tensor,filters = 1,kernel_size = 1, activation = "sigmoid")

unet_model <- keras_model(inputs = input_tensor, outputs = unet_tensor)



vgg16_feat_extr <- application_vgg16(weights = "imagenet", include_top = FALSE, input_shape = c (448,448,3))
unet_tensor <- vgg16_feat_extr$layers[[15]]$output 
unet_tensor <- layer_conv_2d(unet_tensor, filters = 1024, kernel_size = 3, padding = "same", activation = "relu")
unet_tensor <- layer_conv_2d(unet_tensor, filters = 1024, kernel_size = 3, padding = "same", activation = "relu")
unet_tensor <- layer_conv_2d_transpose(unet_tensor, filters = 512, kernel_size = 2, strides = 2, padding = "same")
unet_tensor <- layer_concatenate(list(vgg16_feat_extr$layers[[14]]$output, unet_tensor))
unet_tensor <- layer_conv_2d(unet_tensor, filters = 512, kernel_size = 3, padding = "same", activation = "relu")
unet_tensor <- layer_conv_2d(unet_tensor, filters = 512, kernel_size = 3, padding = "same", activation = "relu")
unet_tensor <- layer_conv_2d_transpose(unet_tensor, filters = 256, kernel_size = 2, strides = 2, padding = "same")
unet_tensor <- layer_concatenate(list(vgg16_feat_extr$layers[[10]]$output, unet_tensor))
unet_tensor <- layer_conv_2d(unet_tensor,filters = 256, kernel_size = 3, padding = "same", activation = "relu")
unet_tensor <- layer_conv_2d(unet_tensor,filters = 256, kernel_size = 3, padding = "same", activation = "relu")
unet_tensor <- layer_conv_2d_transpose(unet_tensor, filters = 128, kernel_size = 2, strides = 2, padding = "same")
unet_tensor <- layer_concatenate(list(vgg16_feat_extr$layers[[6]]$output, unet_tensor))
unet_tensor <- layer_conv_2d(unet_tensor, filters = 128, kernel_size = 3, padding = "same", activation = "relu")
unet_tensor <- layer_conv_2d(unet_tensor, filters = 128, kernel_size = 3, padding = "same", activation = "relu")
unet_tensor <- layer_conv_2d_transpose(unet_tensor, filters = 64, kernel_size = 2, strides = 2, padding = "same")
unet_tensor <- layer_concatenate(list(vgg16_feat_extr$layers[[3]]$output, unet_tensor))
unet_tensor <- layer_conv_2d(unet_tensor, filters = 64, kernel_size = 3, padding = "same", activation = "relu")
unet_tensor <- layer_conv_2d(unet_tensor, filters = 64, kernel_size = 3, padding = "same", activation = "relu")
unet_tensor <- layer_conv_2d(unet_tensor, filters = 1, kernel_size = 1, activation = "sigmoid")
pretrained_unet <- keras_model(inputs = vgg16_feat_extr$input, outputs = unet_tensor)


spectral_augmentation <- function(img) {
  img <- tf$image$random_brightness(img, max_delta = 0.3) 
  img <- tf$image$random_contrast(img, lower = 0.8, upper = 1.2)
  img <- tf$image$random_saturation(img, lower = 0.8, upper = 1.2) 
  img <- tf$clip_by_value(img,0, 1) 
}


dl_prepare_data <- function(files=NULL, train, predict=FALSE, subsets_path=NULL, model_input_shape = c(448,448), batch_size = 10L) {
  
  if (!predict){
    
    spectral_augmentation <- function(img) {
      img <- tf$image$random_brightness(img, max_delta = 0.3)
      img <- tf$image$random_contrast(img, lower = 0.8, upper = 1.1)
      img <- tf$image$random_saturation(img, lower = 0.8, upper = 1.1)
      img <- tf$clip_by_value(img, 0, 1)
    }
    
    dataset <- tensor_slices_dataset(files)
    dataset <- 
      dataset_map(dataset, function(.x) 
        list_modify(.x,img = tf$image$decode_jpeg(tf$io$read_file(.x$img)),
                    mask = tf$image$decode_jpeg(tf$io$read_file(.x$mask)))) 
    dataset <- 
      dataset_map(dataset, function(.x) 
        list_modify(.x, img = tf$image$convert_image_dtype(.x$img, dtype = tf$float32),
                    mask = tf$image$convert_image_dtype(.x$mask, dtype = tf$float32)))
    dataset <- 
      dataset_map(dataset, function(.x) 
        list_modify(.x, img = tf$image$resize(.x$img, size = shape(model_input_shape[1], model_input_shape[2])),
                    mask = tf$image$resize(.x$mask, size = shape(model_input_shape[1], model_input_shape[2]))))
    
    if (train) {
      augmentation <- 
        dataset_map(dataset, function(.x) 
          list_modify(.x, img = spectral_augmentation(.x$img)))
      augmentation <- 
        dataset_map(augmentation, function(.x) 
          list_modify(.x, img = tf$image$flip_left_right(.x$img),
                      mask = tf$image$flip_left_right(.x$mask)))
      dataset_augmented <- dataset_concatenate(dataset,augmentation)
      augmentation <- 
        dataset_map(dataset, function(.x) 
          list_modify(.x, img = spectral_augmentation(.x$img)))
      augmentation <- 
        dataset_map(augmentation, function(.x) 
          list_modify(.x, img = tf$image$flip_up_down(.x$img),
                      mask = tf$image$flip_up_down(.x$mask)))
      dataset_augmented <- dataset_concatenate(dataset_augmented,augmentation)
      augmentation <- 
        dataset_map(dataset, function(.x) 
          list_modify(.x, img = spectral_augmentation(.x$img)))
      augmentation <- 
        dataset_map(augmentation, function(.x) 
          list_modify(.x, img = tf$image$flip_left_right(.x$img),
                      mask = tf$image$flip_left_right(.x$mask)))
      augmentation <- 
        dataset_map(augmentation, function(.x) 
          list_modify(.x, img = tf$image$flip_up_down(.x$img),
                      mask = tf$image$flip_up_down(.x$mask)))
      dataset_augmented <- dataset_concatenate(dataset_augmented,augmentation)
    }
    
    
    if (train) {
      dataset <- dataset_shuffle(dataset_augmented, buffer_size = batch_size*128)
    }
  
    dataset <- dataset_batch(dataset, batch_size)
    dataset <-  dataset_map(dataset, unname) 
  }else{
    
    o <- order(as.numeric(tools::file_path_sans_ext(basename(list.files(subsets_path)))))
    subset_list <- list.files(subsets_path, full.names = T)[o]
    
    dataset <- tensor_slices_dataset(subset_list)
    
    dataset <- 
      dataset_map(dataset, function(.x) 
        tf$image$decode_jpeg(tf$io$read_file(.x))) 
    
    dataset <- 
      dataset_map(dataset, function(.x) 
        tf$image$convert_image_dtype(.x, dtype = tf$float32)) 
    
    dataset <- 
      dataset_map(dataset, function(.x) 
        tf$image$resize(.x, size = shape(model_input_shape[1], model_input_shape[2]))) 
    
    dataset <- dataset_batch(dataset, batch_size)
    dataset <-  dataset_map(dataset, unname)
    
  }
  
}



# Einladen der Trainingsdaten

files <- data.frame(
  img = list.files("Data/Subsets_448/Slices_Berlin", full.names = TRUE, pattern = "*.jpg"),
  mask = list.files("Data/Subsets_448/Slices_Berlin_Mask", full.names = TRUE, pattern = "*.jpg")
)
files <- initial_split(files, prop = 0.8)
training_dataset <- dl_prepare_data(training(files),train = TRUE,model_input_shape = c(448,448),batch_size = 10L)
validation_dataset <- dl_prepare_data(testing(files),train = FALSE,model_input_shape = c(448,448),batch_size = 10L)

training_tensors <- training_dataset%>%as_iterator()%>%iterate()

# Training des Unets
compile(
  pretrained_unet,
  optimizer = optimizer_rmsprop(lr = 1e-5),
  loss = "binary_crossentropy",
  metrics = c(metric_binary_accuracy)
)


diagnostics <- fit(pretrained_unet,
                   training_dataset,
                   epochs = 15,
                   validation_data = validation_dataset)

plot(diagnostics)

save_model_hdf5(pretrained_unet,filepath = "Unets/pretrained_unet_versuch4")

pretrained_unet <- load_model_hdf5("Unets/pretrained_unet_versuch4")


# Vergleich Maske/Satellitenbild/Vorhersage
sample <- floor(runif(n = 1,min = 1,max = 10))
img_path <- as.character(testing(files)[[sample,1]])
mask_path <- as.character(testing(files)[[sample,2]])
img <- magick::image_read(img_path)
mask <- magick::image_read(mask_path)
pred <- magick::image_read(as.raster(predict(object = pretrained_unet,validation_dataset)[sample,,,]))

out <- magick::image_append(c(
  magick::image_append(mask, stack = TRUE),
  magick::image_append(img, stack = TRUE), 
  magick::image_append(pred, stack = TRUE)
)
)

plot(out)


# Einladen der Validierungsdaten
test_dataset <- dl_prepare_data(train = F,predict = T,subsets_path="Data/Subsets_448/Slices_Muenster/",model_input_shape = c(448,448),batch_size = 5L)

system.time(predictions <- predict(pretrained_unet,test_dataset))



plot_layer_activations <- function(img_path, model, activations_layers,channels){
  
  model_input_size <- c(model$input_shape[[2]], model$input_shape[[3]]) 
  
  img <- image_load(img_path, target_size =  model_input_size) %>%
    image_to_array() %>%
    array_reshape(dim = c(1, model_input_size[1], model_input_size[2], 3)) %>%
    imagenet_preprocess_input()
  
  layer_outputs <- lapply(model$layers[activations_layers], function(layer) layer$output)
  activation_model <- keras_model(inputs = model$input, outputs = layer_outputs)
  activations <- predict(activation_model,img)
  if(!is.list(activations)){
    activations <- list(activations)
  }
  
  plot_channel <- function(channel,layer_name,channel_name) {
    rotate <- function(x) t(apply(x, 2, rev))
    image(rotate(channel), axes = FALSE, asp = 1,
          col = terrain.colors(12),main=paste("layer:",layer_name,"channel:",channel_name))
  }
  
  for (i in 1:length(activations)) {
    layer_activation <- activations[[i]]
    layer_name <- model$layers[[activations_layers[i]]]$name
    n_features <- dim(layer_activation)[[4]]
    for (c in channels){
      
      channel_image <- layer_activation[1,,,c]
      plot_channel(channel_image,layer_name,c)
      
    }
  } 
  
}



par(mfrow=c(1,1))
plot(read_stars("Data/Subsets_448/Slices_Muenster/M_01.jpg"),rgb=c(1,2,3))
plot(read_stars("Data/Subsets_448/Slices_Muenster/M_02.jpg"),rgb=c(1,2,3))
plot(read_stars("Data/Subsets_448/Slices_Muenster/M_03.jpg"),rgb=c(1,2,3))
plot(read_stars("Data/Subsets_448/Slices_Muenster/M_04.jpg"),rgb=c(1,2,3))
plot(read_stars("Data/Subsets_448/Slices_Muenster/M_05.jpg"),rgb=c(1,2,3))
plot(read_stars("Data/Subsets_448/Slices_Muenster/M_06.jpg"),rgb=c(1,2,3))


par(mfrow=c(3,4),mar=c(1,1,1,1),cex=0.5)
plot_layer_activations(img_path = "Data/Subsets_448/Slices_Muenster/M_01.jpg", model=pretrained_unet ,activations_layers = c(2,3,5,6,8,9,10,12,13,14), channels = 1:4)


