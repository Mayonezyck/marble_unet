from model import *
from data import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
k_value = 10
n_img = 526
kfolderGenerator(k_value,'inputs','Labels')
i = 0
while i < k_value:
#for i in range(k_value):
    print('Now working on folder -'+str(i))
    test_num = math.floor(n_img/k_value)
    if i == k_value-1:
        test_num += n_img%k_value
    kpath = 'Kfolder'
    #myGene = trainGenerator(3,os.path.join(kpath,str(i),'train'),'image','labels',os.path.join(kpath,str(i),'train','aug'),data_gen_args,)
    myGene = trainGenerator(3,os.path.join(kpath,str(i),'train'),'image','labels',None,data_gen_args)
    model = unet()
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    #model_checkpoint = ModelCheckpoint(os.path.join(kpath,str(i),'unet_marble_'+str(i)+'_.hdf5'), monitor='loss',verbose=1, save_best_only=True)
    #model.fit(myGene,steps_per_epoch=300,epochs=4,callbacks=[model_checkpoint])
    history = model.fit(myGene,steps_per_epoch=300,epochs=20,callbacks=[early_stop])
    length = len(history.history['loss'])
    testGene = testGenerator(os.path.join(kpath,str(i),'test'),num_image = test_num)
    results = model.predict(testGene,test_num,verbose=1)
    saveResult(i,os.path.join(kpath,str(i),'test'),results)
    print(history.history['loss'][length-1])
    if (history.history['loss'][length-1])>0.1:
    	i -=1
    else:
    	model.save(os.path.join(kpath,str(i),'unet_marble.hdf5'))
    i += 1
