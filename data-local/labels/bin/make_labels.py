import os


def make_labels(dataset):
	cwd = os.getcwd()

	# make train labels
	train_label_dir = os.path.join(cwd[:cwd.find('bin')], dataset, 'train')
	if not os.path.exists(train_label_dir):
		os.makedirs(train_label_dir)
	filename = os.path.join(train_label_dir, 'train_labels.txt')
	if dataset.startswith('cifar'):
		train_img_dir = os.path.join(cwd[:cwd.find('labels')], 'images', 'cifar', dataset, 'by-image', 'train')
	else:
		train_img_dir = os.path.join(cwd[:cwd.find('labels')], 'images', dataset, 'train')

	class_names = sorted(os.listdir(train_img_dir))
	with open(filename, 'w') as f:
		for c in class_names:
			for filename in os.listdir(os.path.join(train_img_dir, c)):
				f.write(f"{filename} {c}" + '\n')

	# make eval labels
	for eval_type in ['val', 'test']:
		val_label_dir = os.path.join(cwd[:cwd.find('bin')], dataset, eval_type)
		if not os.path.exists(val_label_dir):
			os.makedirs(val_label_dir)
		if dataset.startswith('cifar'):
			val_img_dir = os.path.join(cwd[:cwd.find('labels')], 'images', 'cifar', dataset, 'by-image', eval_type)
		else:
			val_img_dir = os.path.join(cwd[:cwd.find('labels')], 'images', dataset, 'test')

		filename = os.path.join(val_label_dir, f'{eval_type}_labels.txt')
		with open(filename, 'w') as f:
			for c in class_names:
				for filename in os.listdir(os.path.join(val_img_dir, c)):
					f.write(f"{filename} {c}" + '\n')


def main():
	for dataset in ['cifar10', 'cifar100', 'miniimagenet']:
		make_labels(dataset)


if __name__ == '__main__':
	main()
