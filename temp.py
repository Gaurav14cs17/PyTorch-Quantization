 if not os.path.isdir(save_folder):
            os.makedirs(save_folder)

        save_name = save_folder +  os.path.basename(image_name).split('.')[0] + ".txt"
        dirname = os.path.dirname(save_name)
        print(dirname ,  os.path.basename(image_name).split('.')[0] )

        with open(save_name, "w") as fd:
            file_name = os.path.basename(save_name)[:-4] + "\n"
            bboxs_num = str(len(bboxs)) + "\n"
            fd.write(file_name)
            fd.write(bboxs_num)
            for i, box in enumerate(bboxs):
                x = int(box[0])
                y = int(box[1])
                w = int(box[2]) - int(box[0])
                h = int(box[3]) - int(box[1])
                confidence = str(scores[i].cpu().numpy())
                line = str(x) + " " + str(y) + " " + str(w) + " " + str(h) + " " + confidence + " \n"
                fd.write(line)
