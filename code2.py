with torch.no_grad():
  train_preds = model(train_data)
  val_preds = model(val_data)
  all_preds = model(data)
np.save('all_compressed_v2.npy', all_preds.numpy())
np.save('all_S8_v2.npy', targets.numpy())
# print(train_targets[:, 0], train_preds[:, 0])
plt.scatter(train_targets[:, 0], train_preds[:, 0])
plt.ylabel('Predicted S8')
plt.xlabel('True S8')
# plt.ylim(0.1, 0.4)
plt.show()
plt.scatter(val_targets[:, 0], val_preds[:, 0])
plt.ylabel('Predicted S8')
plt.xlabel('True S8')
# plt.ylim(0.1, 0.4)
plt.show()
plt.scatter(targets[:, 0], all_preds[:, 0])
plt.ylabel('Predicted S8')
plt.xlabel('True S8')
# plt.ylim(0.1, 0.4)
plt.show()



coeff_means = np.mean(data, axis=0)
coeff_std = np.std(data, axis=0)
data = (data - coeff_means[None, :]) / coeff_std[None, :]







