<!-- Output copied to clipboard! -->

<!-----
NEW: Check the "Suppress top comment" option to remove this info from the output.

Conversion time: 3.242 seconds.


Using this Markdown file:

1. Paste this output into your source file.
2. See the notes and action items below regarding this conversion run.
3. Check the rendered output (headings, lists, code blocks, tables) for proper
   formatting and use a linkchecker before you publish this page.

Conversion notes:

* Docs to Markdown version 1.0β29
* Sun Feb 28 2021 09:21:54 GMT-0800 (PST)
* Source doc: Noshow Results
* Tables are currently converted to HTML tables.
----->


Working on the no-show dataset ([https://www.kaggle.com/joniarroba/noshowappointments](https://www.kaggle.com/joniarroba/noshowappointments)), a dataset of patient appointments, we attempt to predict whether or not a patient will show up for their appointment. Only about ¼ of the patients are no-shows, and in this repo, we show that by generating more no-shows, we can improve the performance of patient no-show classifiers. See the results below:

Original dataset results:


<table>
  <tr>
   <td>Model
   </td>
   <td><strong>Accuracy</strong>
   </td>
   <td><strong>AUC</strong>
   </td>
   <td><strong>Recall</strong>
   </td>
   <td><strong>Prec.</strong>
   </td>
   <td><strong>F1</strong>
   </td>
   <td><strong>Kappa</strong>
   </td>
   <td><strong>MCC</strong>
   </td>
   <td><strong>TT (Sec)</strong>
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td><strong>catboost</strong>
   </td>
   <td>CatBoost Classifier
   </td>
   <td>0.8026
   </td>
   <td>0.7461
   </td>
   <td>0.0778
   </td>
   <td>0.5843
   </td>
   <td>0.1372
   </td>
   <td>0.0942
   </td>
   <td>0.1582
   </td>
   <td>14.866
   </td>
  </tr>
  <tr>
   <td><strong>lightgbm</strong>
   </td>
   <td>Light Gradient Boosting Machine
   </td>
   <td>0.8015
   </td>
   <td>0.7433
   </td>
   <td>0.0376
   </td>
   <td>0.6444
   </td>
   <td>0.0711
   </td>
   <td>0.05
   </td>
   <td>0.1204
   </td>
   <td>39.915
   </td>
  </tr>
  <tr>
   <td><strong>xgboost</strong>
   </td>
   <td>Extreme Gradient Boosting
   </td>
   <td>0.8003
   </td>
   <td>0.7431
   </td>
   <td>0.092
   </td>
   <td>0.5332
   </td>
   <td>0.1569
   </td>
   <td>0.1035
   </td>
   <td>0.1567
   </td>
   <td>6.864
   </td>
  </tr>
  <tr>
   <td><strong>rf</strong>
   </td>
   <td>Random Forest Classifier
   </td>
   <td>0.8022
   </td>
   <td>0.7411
   </td>
   <td>0.1601
   </td>
   <td>0.5339
   </td>
   <td>0.2463
   </td>
   <td>0.169
   </td>
   <td>0.21
   </td>
   <td>4.068
   </td>
  </tr>
  <tr>
   <td><strong>gbc</strong>
   </td>
   <td>Gradient Boosting Classifier
   </td>
   <td>0.7984
   </td>
   <td>0.7332
   </td>
   <td>0.0067
   </td>
   <td>0.6078
   </td>
   <td>0.0132
   </td>
   <td>0.0086
   </td>
   <td>0.0463
   </td>
   <td>3.843
   </td>
  </tr>
  <tr>
   <td><strong>ada</strong>
   </td>
   <td>Ada Boost Classifier
   </td>
   <td>0.7976
   </td>
   <td>0.7282
   </td>
   <td>0.0168
   </td>
   <td>0.463
   </td>
   <td>0.0323
   </td>
   <td>0.0186
   </td>
   <td>0.0557
   </td>
   <td>0.924
   </td>
  </tr>
  <tr>
   <td><strong>et</strong>
   </td>
   <td>Extra Trees Classifier
   </td>
   <td>0.7905
   </td>
   <td>0.726
   </td>
   <td>0.1991
   </td>
   <td>0.4573
   </td>
   <td>0.2773
   </td>
   <td>0.1765
   </td>
   <td>0.1974
   </td>
   <td>6.206
   </td>
  </tr>
  <tr>
   <td><strong>lda</strong>
   </td>
   <td>Linear Discriminant Analysis
   </td>
   <td>0.791
   </td>
   <td>0.681
   </td>
   <td>0.0436
   </td>
   <td>0.3569
   </td>
   <td>0.0776
   </td>
   <td>0.0353
   </td>
   <td>0.0613
   </td>
   <td>1.368
   </td>
  </tr>
  <tr>
   <td><strong>lr</strong>
   </td>
   <td>Logistic Regression
   </td>
   <td>0.7954
   </td>
   <td>0.6784
   </td>
   <td>0.025
   </td>
   <td>0.398
   </td>
   <td>0.0471
   </td>
   <td>0.0236
   </td>
   <td>0.0552
   </td>
   <td>5.518
   </td>
  </tr>
  <tr>
   <td><strong>knn</strong>
   </td>
   <td>K Neighbors Classifier
   </td>
   <td>0.7778
   </td>
   <td>0.6744
   </td>
   <td>0.2076
   </td>
   <td>0.403
   </td>
   <td>0.2739
   </td>
   <td>0.1583
   </td>
   <td>0.1705
   </td>
   <td>9.431
   </td>
  </tr>
  <tr>
   <td><strong>nb</strong>
   </td>
   <td>Naive Bayes
   </td>
   <td>0.2345
   </td>
   <td>0.5988
   </td>
   <td>0.9611
   </td>
   <td>0.204
   </td>
   <td>0.3365
   </td>
   <td>0.005
   </td>
   <td>0.0206
   </td>
   <td>0.087
   </td>
  </tr>
  <tr>
   <td><strong>dt</strong>
   </td>
   <td>Decision Tree Classifier
   </td>
   <td>0.7344
   </td>
   <td>0.5862
   </td>
   <td>0.3361
   </td>
   <td>0.3404
   </td>
   <td>0.3382
   </td>
   <td>0.1721
   </td>
   <td>0.1721
   </td>
   <td>0.482
   </td>
  </tr>
  <tr>
   <td><strong>qda</strong>
   </td>
   <td>Quadratic Discriminant Analysis
   </td>
   <td>0.5235
   </td>
   <td>0.5083
   </td>
   <td>0.4827
   </td>
   <td>0.2066
   </td>
   <td>0.2772
   </td>
   <td>0.0098
   </td>
   <td>0.0144
   </td>
   <td>7.784
   </td>
  </tr>
  <tr>
   <td><strong>svm</strong>
   </td>
   <td>SVM - Linear Kernel
   </td>
   <td>0.7981
   </td>
   <td>0
   </td>
   <td>0
   </td>
   <td>0
   </td>
   <td>0
   </td>
   <td>0
   </td>
   <td>0
   </td>
   <td>0.218
   </td>
  </tr>
  <tr>
   <td><strong>ridge</strong>
   </td>
   <td>Ridge Classifier
   </td>
   <td>0.7976
   </td>
   <td>0
   </td>
   <td>0.0092
   </td>
   <td>0.4512
   </td>
   <td>0.018
   </td>
   <td>0.01
   </td>
   <td>0.0396
   </td>
   <td>0.077
   </td>
  </tr>
</table>


Results with synthetic dataset:


<table>
  <tr>
   <td>Model
   </td>
   <td><strong>Accuracy</strong>
   </td>
   <td><strong>AUC</strong>
   </td>
   <td><strong>Recall</strong>
   </td>
   <td><strong>Prec.</strong>
   </td>
   <td><strong>F1</strong>
   </td>
   <td><strong>Kappa</strong>
   </td>
   <td><strong>MCC</strong>
   </td>
   <td><strong>TT (Sec)</strong>
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td><strong>catboost</strong>
   </td>
   <td>CatBoost Classifier
   </td>
   <td>0.86
   </td>
   <td>0.9333
   </td>
   <td>0.7607
   </td>
   <td>0.9283
   </td>
   <td>0.7721
   </td>
   <td>0.7213
   </td>
   <td>0.7438
   </td>
   <td>20.435
   </td>
  </tr>
  <tr>
   <td><strong>xgboost</strong>
   </td>
   <td>Extreme Gradient Boosting
   </td>
   <td>0.8552
   </td>
   <td>0.93
   </td>
   <td>0.7629
   </td>
   <td>0.9094
   </td>
   <td>0.7726
   </td>
   <td>0.7116
   </td>
   <td>0.7324
   </td>
   <td>8.432
   </td>
  </tr>
  <tr>
   <td><strong>rf</strong>
   </td>
   <td>Random Forest Classifier
   </td>
   <td>0.8521
   </td>
   <td>0.9292
   </td>
   <td>0.7843
   </td>
   <td>0.8887
   </td>
   <td>0.7896
   </td>
   <td>0.7052
   </td>
   <td>0.7262
   </td>
   <td>6.809
   </td>
  </tr>
  <tr>
   <td><strong>lightgbm</strong>
   </td>
   <td>Light Gradient Boosting Machine
   </td>
   <td>0.8534
   </td>
   <td>0.9273
   </td>
   <td>0.7582
   </td>
   <td>0.9005
   </td>
   <td>0.7707
   </td>
   <td>0.7079
   </td>
   <td>0.7257
   </td>
   <td>1.061
   </td>
  </tr>
  <tr>
   <td><strong>et</strong>
   </td>
   <td>Extra Trees Classifier
   </td>
   <td>0.8393
   </td>
   <td>0.9183
   </td>
   <td>0.8018
   </td>
   <td>0.8482
   </td>
   <td>0.7943
   </td>
   <td>0.6794
   </td>
   <td>0.6974
   </td>
   <td>11.339
   </td>
  </tr>
  <tr>
   <td><strong>gbc</strong>
   </td>
   <td>Gradient Boosting Classifier
   </td>
   <td>0.8092
   </td>
   <td>0.9063
   </td>
   <td>0.7941
   </td>
   <td>0.8015
   </td>
   <td>0.7782
   </td>
   <td>0.619
   </td>
   <td>0.6325
   </td>
   <td>6.779
   </td>
  </tr>
  <tr>
   <td><strong>knn</strong>
   </td>
   <td>K Neighbors Classifier
   </td>
   <td>0.8018
   </td>
   <td>0.8786
   </td>
   <td>0.7211
   </td>
   <td>0.8423
   </td>
   <td>0.7494
   </td>
   <td>0.6045
   </td>
   <td>0.6188
   </td>
   <td>22.901
   </td>
  </tr>
  <tr>
   <td><strong>ada</strong>
   </td>
   <td>Ada Boost Classifier
   </td>
   <td>0.7615
   </td>
   <td>0.8589
   </td>
   <td>0.776
   </td>
   <td>0.7462
   </td>
   <td>0.7521
   </td>
   <td>0.5231
   </td>
   <td>0.5322
   </td>
   <td>1.469
   </td>
  </tr>
  <tr>
   <td><strong>lr</strong>
   </td>
   <td>Logistic Regression
   </td>
   <td>0.7455
   </td>
   <td>0.8153
   </td>
   <td>0.7272
   </td>
   <td>0.7437
   </td>
   <td>0.7262
   </td>
   <td>0.4913
   </td>
   <td>0.4969
   </td>
   <td>5.689
   </td>
  </tr>
  <tr>
   <td><strong>lda</strong>
   </td>
   <td>Linear Discriminant Analysis
   </td>
   <td>0.7457
   </td>
   <td>0.8148
   </td>
   <td>0.7273
   </td>
   <td>0.7446
   </td>
   <td>0.727
   </td>
   <td>0.4918
   </td>
   <td>0.4973
   </td>
   <td>2.22
   </td>
  </tr>
  <tr>
   <td><strong>dt</strong>
   </td>
   <td>Decision Tree Classifier
   </td>
   <td>0.8098
   </td>
   <td>0.8101
   </td>
   <td>0.796
   </td>
   <td>0.8
   </td>
   <td>0.7767
   </td>
   <td>0.6201
   </td>
   <td>0.6349
   </td>
   <td>0.719
   </td>
  </tr>
  <tr>
   <td><strong>nb</strong>
   </td>
   <td>Naive Bayes
   </td>
   <td>0.5981
   </td>
   <td>0.7097
   </td>
   <td>0.2909
   </td>
   <td>0.7307
   </td>
   <td>0.4108
   </td>
   <td>0.1992
   </td>
   <td>0.2414
   </td>
   <td>0.125
   </td>
  </tr>
  <tr>
   <td><strong>qda</strong>
   </td>
   <td>Quadratic Discriminant Analysis
   </td>
   <td>0.5032
   </td>
   <td>0.5046
   </td>
   <td>0.2655
   </td>
   <td>0.5167
   </td>
   <td>0.3194
   </td>
   <td>0.0092
   </td>
   <td>0.0121
   </td>
   <td>2.023
   </td>
  </tr>
  <tr>
   <td><strong>svm</strong>
   </td>
   <td>SVM - Linear Kernel
   </td>
   <td>0.7413
   </td>
   <td>0
   </td>
   <td>0.7165
   </td>
   <td>0.745
   </td>
   <td>0.7236
   </td>
   <td>0.483
   </td>
   <td>0.4875
   </td>
   <td>0.374
   </td>
  </tr>
  <tr>
   <td><strong>ridge</strong>
   </td>
   <td>Ridge Classifier
   </td>
   <td>0.7457
   </td>
   <td>0
   </td>
   <td>0.7273
   </td>
   <td>0.7446
   </td>
   <td>0.727
   </td>
   <td>0.4918
   </td>
   <td>0.4973
   </td>
   <td>0.088
   </td>
  </tr>
</table>

