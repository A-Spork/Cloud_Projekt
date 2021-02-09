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
library(jpeg)
library(mapview)

setwd("./Cloud")

read_tif <- function(f,mask=FALSE) {
  out = array(NA)
  out = unclass(read_stars(f))[[1]]
  if(mask==T){
    dim(out) <- c(dim(out),1)
  }
  return(out)
}

files_train <- data.frame(img="./Train_Berlin.tif", mask="./Mask_Berlin.tif")
files_val <- data.frame(img="./Val_Muenster.tif", mask="./Mask_Muenster.tif")

files_train$img <- lapply(files_train$img, read_tif)
files_train$img <- lapply(files_train$img, function(x){x/10000}) 
files_train$mask <- lapply(files_train$mask, read_tif, TRUE)
files_val$img <- lapply(files_val$img, read_tif)
files_val$img <- lapply(files_val$img, function(x){x/10000})
files_val$mask <- lapply(files_val$mask, read_tif,TRUE)

dl_prepare_data_tif <- function(files, train, predict=FALSE, subsets_path=NULL, model_input_shape = c(448,448), batch_size = 10L) {
  
  if (!predict){
    
    #function for random change of saturation,brightness and hue, will be used as part of the augmentation
    spectral_augmentation <- function(img) {
      img %>% 
        tf$image$random_brightness(max_delta = 0.3) %>% 
        tf$image$random_contrast(lower = 0.5, upper = 0.7) %>% 
        #tf$image$random_saturation(lower = 0.5, upper = 0.7) %>%  --> not supported for >3 bands - you can uncomment in case you use only 3band images
        # make sure we still are between 0 and 1
        tf$clip_by_value(0, 1) 
    }
    
    
    #create a tf_dataset from the first two coloumns of data.frame (ignoring area number used for splitting during data preparation),
    #right now still containing only paths to images 
    dataset <- tensor_slices_dataset(files[,1:2])
    
    
    #the following (replacing tf$image$decode_jpeg by the custom read_tif function) doesn't work, since read_tif cannot be used with dataset_map -> dl_prepare_data_tif therefore expects a data.frame with arrays (i.e. images already loaded)
    #dataset <- dataset_map(dataset, function(.x) list_modify(.x,
    #                                                         img = read_tif(.x$img)/10000,
    #                                                         mask = read_tif(.x$mask)#[1,,,][,,1,drop=FALSE]
    #)) 
    
    
    #convert to float32:
    #for each record in dataset, both its list items are modyfied by the result of applying convert_image_dtype to them
    dataset <- dataset_map(dataset, function(.x) list_modify(.x,
                                                             img = tf$image$convert_image_dtype(.x$img, dtype = tf$float64),
                                                             mask = tf$image$convert_image_dtype(.x$mask, dtype = tf$float64)
    )) 
    
    #resize:
    #for each record in dataset, both its list items are modified by the results of applying resize to them 
    dataset <- 
      dataset_map(dataset, function(.x) 
        list_modify(.x, img = tf$image$resize(.x$img, size = shape(model_input_shape[1], model_input_shape[2])),
                    mask = tf$image$resize(.x$mask, size = shape(model_input_shape[1], model_input_shape[2]))))
    
    
    # data augmentation performed on training set only
    if (train) {
      
      #augmentation 1: flip left right, including random change of saturation, brightness and contrast
      
      #for each record in dataset, only the img item is modified by the result of applying spectral_augmentation to it
      augmentation <- dataset_map(dataset, function(.x) list_modify(.x,
                                                                    img = spectral_augmentation(.x$img)
      ))
      #...as opposed to this, flipping is applied to img and mask of each record
      augmentation <- dataset_map(augmentation, function(.x) list_modify(.x,
                                                                         img = tf$image$flip_left_right(.x$img),
                                                                         mask = tf$image$flip_left_right(.x$mask)
      ))
      dataset_augmented <- dataset_concatenate(dataset,augmentation)
      
      #augmentation 2: flip up down, including random change of saturation, brightness and contrast
      augmentation <- dataset_map(dataset, function(.x) list_modify(.x,
                                                                    img = spectral_augmentation(.x$img)
      ))
      augmentation <- dataset_map(augmentation, function(.x) list_modify(.x,
                                                                         img = tf$image$flip_up_down(.x$img),
                                                                         mask = tf$image$flip_up_down(.x$mask)
      ))
      dataset_augmented <- dataset_concatenate(dataset_augmented,augmentation)
      
      #augmentation 3: flip left right AND up down, including random change of saturation, brightness and contrast
      augmentation <- dataset_map(dataset, function(.x) list_modify(.x,
                                                                    img = spectral_augmentation(.x$img)
      ))
      augmentation <- dataset_map(augmentation, function(.x) list_modify(.x,
                                                                         img = tf$image$flip_left_right(.x$img),
                                                                         mask = tf$image$flip_left_right(.x$mask)
      ))
      augmentation <- dataset_map(augmentation, function(.x) list_modify(.x,
                                                                         img = tf$image$flip_up_down(.x$img),
                                                                         mask = tf$image$flip_up_down(.x$mask)
      ))
      dataset_augmented <- dataset_concatenate(dataset_augmented,augmentation)
      
    }
    
    # shuffling on training set only
    if (train) {
      dataset <- dataset_shuffle(dataset_augmented, buffer_size = batch_size*128)
    }
    
    # train in batches; batch size might need to be adapted depending on
    # available memory
    dataset <- dataset_batch(dataset, batch_size)
    
    # output needs to be unnamed
    dataset <-  dataset_map(dataset, unname) 
    
  }else{
    #make sure subsets are read in in correct order so that they can later be reassambled correctly
    #needs files to be named accordingly (only number)
    o <- order(as.numeric(tools::file_path_sans_ext(basename(list.files(subsets_path)))))
    subset_list <- list.files(subsets_path, full.names = T)[o]
    
    dataset <- tensor_slices_dataset(subset_list)
    #dataset <- dataset_map(dataset, function(.x) tf$image$decode_jpeg(tf$io$read_file(.x))) 
    dataset <- dataset_map(dataset, function(.x) tf$image$convert_image_dtype(.x, dtype = tf$float32)) 
    dataset <- dataset_map(dataset, function(.x) tf$image$resize(.x, size = shape(model_input_shape[1], model_input_shape[2]))) 
    dataset <- dataset_batch(dataset, batch_size)
    dataset <-  dataset_map(dataset, unname)
    
  }
  
}

dl_subsets <- function(inputrst, targetsize, targetdir, targetname="", img_info_only = FALSE, is_mask = FALSE){
  require(jpeg)
  require(raster)
  targetsizeX <- targetsize[1]
  targetsizeY <- targetsize[2]
  inputX <- ncol(inputrst)
  inputY <- nrow(inputrst)
  
  while(inputX%%targetsizeX!=0){
    inputX = inputX-1  
  }
  while(inputY%%targetsizeY!=0){
    inputY = inputY-1    
  }
  
  diffX <- ncol(inputrst)-inputX
  diffY <- nrow(inputrst)-inputY
  
  newXmin <- floor(diffX/2)
  newXmax <- ncol(inputrst)-ceiling(diffX/2)-1
  newYmin <- floor(diffY/2)
  newYmax <- nrow(inputrst)-ceiling(diffY/2)-1
  rst_cropped <- suppressMessages(crop(inputrst, extent(inputrst,newYmin,newYmax,newXmin,newXmax)))
  agg <- suppressMessages(aggregate(rst_cropped[[1]],c(targetsizeX,targetsizeY)))
  agg[]    <- suppressMessages(1:ncell(agg))
  agg_poly <- suppressMessages(rasterToPolygons(agg))
  names(agg_poly) <- "polis"
  
  pb <- txtProgressBar(min = 0, max = ncell(agg), style = 3)
  for(i in 1:ncell(agg)) {
    
    setTxtProgressBar(pb, i)
    e1  <- extent(agg_poly[agg_poly$polis==i,])
    
    subs <- suppressMessages(crop(rst_cropped,e1))
    
    if(is_mask==FALSE){
      
      subs <- suppressMessages((subs-cellStats(subs,"min"))/(cellStats(subs,"max")-cellStats(subs,"min")))
    } 
    
    
    
    writeJPEG(as.array(subs),target = paste0(targetdir,targetname,i,".jpg"),quality = 1)
  }
  close(pb)
  rm(subs,agg,agg_poly)
  gc()
  return(rst_cropped)
  
}

datasubsets_train <- dl_subsets(files_train, targetsize = c(128, 128), targetdir = "./Subsets", targetname="subsets")
datasubsets_val <- dl_subsets(files_val, targetsize = c(128, 128), targetdir = "./Subsets", targetname="subsets")

#training_dataset <- dl_prepare_data_tif(datasubsets_train,train = TRUE,model_input_shape = c(256,256),batch_size = 10L)
#validating_dataset <- dl_prepare_data_tif(datasubsets_val,train = TRUE,model_input_shape = c(256,256),batch_size = 10L)

#training_dataset <- dl_prepare_data_tif(files_train,train = TRUE,model_input_shape = c(256,256),batch_size = 10L)
#validating_dataset <- dl_prepare_data_tif(files_val,train = TRUE,model_input_shape = c(256,256),batch_size = 10L)
