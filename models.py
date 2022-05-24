
import tensorflow as tf 
from tensorflow.keras import layers,optimizers
from sklearn.metrics import accuracy_score,roc_curve,auc


def model_BRCA(input_shape):
    inputs=layers.Input(shape=input_shape)
        
    x=layers.Dense(256)(inputs)
    x=layers.BatchNormalization()(x)
    x=layers.Activation('relu')(x)
    x=layers.Dropout(0.7)(x)

    x=layers.Dense(128)(x)
    x=layers.BatchNormalization()(x)
    x=layers.Activation('relu')(x)
    x=layers.Dropout(0.6)(x)
    
    x=layers.Dense(64)(x)
    x=layers.BatchNormalization()(x)
    x=layers.Activation('relu')(x)




    x=layers.Dense(2,activation='softmax')(x)
    model=tf.keras.models.Model(inputs=inputs,outputs=x)   
    print(model.summary())
    return model

def model_PAAD(input_shape):#BRCA grid search
    inputs=layers.Input(shape=input_shape)
    
 
    
    x=layers.Dense(512)(inputs)
    x=layers.BatchNormalization()(x)
    x=layers.Activation('relu')(x)
    x=layers.Dropout(0.7)(x)

    x=layers.Dense(256)(x)
    x=layers.BatchNormalization()(x)
    x=layers.Activation('relu')(x)
    x=layers.Dropout(0.5)(x)
    
    x=layers.Dense(128)(x)
    x=layers.BatchNormalization()(x)
    x=layers.Activation('relu')(x)




    x=layers.Dense(2,activation='softmax')(x)
    model=tf.keras.models.Model(inputs=inputs,outputs=x)   
    print(model.summary())
    return model

def model_PRAD(input_shape):#BRCA grid search
    inputs=layers.Input(shape=input_shape)
  
    
    x=layers.Dense(128)(inputs)
    x=layers.BatchNormalization()(x)
    x=layers.Activation('relu')(x)
    x=layers.Dropout(0.5)(x)

    x=layers.Dense(128)(x)
    x=layers.BatchNormalization()(x)
    x=layers.Activation('relu')(x)
    x=layers.Dropout(0.8)(x)

    x=layers.Dense(64)(x)
    x=layers.BatchNormalization()(x)
    x=layers.Activation('relu')(x)

    x=layers.Dense(2,activation='softmax')(x)
    model=tf.keras.models.Model(inputs=inputs,outputs=x)   
    print(model.summary())
    return model

def training(model2,train_x,train_lbls,test_x,test_lbls,epochs,batch_size,early_stop,class_w,save_model_dir,learning_rate):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # 텐서플로가 첫 번째 GPU에 1GB 메모리만 할당하도록 제한
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
        except RuntimeError as e:
            # 프로그램 시작시에 가상 장치가 설정되어야만 합니다
            print(e)
    aucss=[]
    train_aucss=[]
    max_auc_score=0
    no_up=0
    opt=tf.keras.optimizers.Adam(lr=learning_rate)
    model2.compile(optimizer=opt,loss='sparse_categorical_crossentropy')
    for epoch in range(epochs):
        model2.fit(train_x,train_lbls,batch_size=batch_size,epochs=1,class_weight = class_w)

        pred_x=model2.predict(train_x)
        preds=[]
        preds_p=[]
        idx_train=0
        for pred in pred_x:
            preds.append(pred.argmax())
            preds_p.append(pred[1])
            idx_train+=1

        train_acc=accuracy_score(list(train_lbls),preds)
        fpr, tpr, thresholds = roc_curve(list(train_lbls), preds_p,pos_label=1)
        train_auc=auc(fpr, tpr)
        print('\n\nepoch: ',epoch+1)
        print('\ntrain')
        print('acc: ',train_acc)
        print('auc: ',train_auc)
        print()
        train_aucss.append(train_auc)
        
        
        
        pred_x=model2.predict(test_x)

        preds=[]
        preds_p=[]
        idx_test=0
        for pred in pred_x:
            preds.append(pred.argmax())
            preds_p.append(pred[1])
            idx_test+=1



        acc=accuracy_score(list(test_lbls),preds)
        fpr, tpr, thresholds = roc_curve(list(test_lbls), preds_p,pos_label=1)
        auc_score=auc(fpr, tpr)

        print('\ntest')
        print('acc: ',acc)
        print('auc: ',auc_score)

        aucss.append(auc_score)
        if max_auc_score ==0 and train_auc>0.80 and train_acc>0.80 :

            max_auc_score=auc_score
            no_up=0
            model2.save(save_model_dir)
        else:
            if auc_score > max_auc_score and train_auc>0.80 and train_acc>0.80:
                max_auc_score = auc_score
                no_up=0
                model2.save(save_model_dir)
            else:
                no_up+=1
            if no_up==early_stop:
                print()
                print('train dataset AUC(when test AUC is max): ',train_aucss[aucss.index(max_auc_score)])
                print('test dataset MAX AUC: ',max_auc_score)
                print()
                break
        if max_auc_score ==0:
            print()
            print('train dataset AUC(when test AUC is max): None')
            print('test dataset MAX AUC: None')
            print()
            print()
        else:
            print()
            print('train dataset AUC(when test AUC is max): ',train_aucss[aucss.index(max_auc_score)])
            print('test dataset MAX AUC: ',max_auc_score)
            print()
            print()
