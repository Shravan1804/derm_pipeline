import re, os

import pandas as pd

from gi.repository import Eog, GObject, Gio

from easygui import choicebox, enterbox


class QuickLabel(GObject.GObject, Eog.WindowActivatable):

    MASTER_VALIDATED_LABEL_CSV = '/home/shravan/deep-learning/data/usz/master_validated_label.csv'
    LABEL_FILENAME = "label.csv"

    ACTION_NEXT_NEW_IMG = "next_unvalidated_image"
    ACTION_CHOOSE_LABEL = "choose_label"
    ACTION_LABEL_SHORTCUT_MAP = {'0': "correct", '9': "tattoo", '8': "identifiable", '7': "unsuitable", '6': "background"}

    dynamic_methods = ""
    for k, v in ACTION_LABEL_SHORTCUT_MAP.items():
        dynamic_methods+='def _label_' + v + '_activated_cb(self, action, param): self._writer("' + v + '")\n\n'
    exec(dynamic_methods, locals())


    window = GObject.property(type=Eog.Window)
    csv_file = None # os.path.expanduser('~')
    label_df = None
    label_choices = set(ACTION_LABEL_SHORTCUT_MAP.values())
    data_loaded = False

    def __init__(self):
        super().__init__()
        self.action_label = {}
        for k, v in self.ACTION_LABEL_SHORTCUT_MAP.items():
            self.action_label[k] = Gio.SimpleAction(name=v)
            self.action_label[k].connect('activate', getattr(self, '_label_' + v +'_activated_cb'))

        self.action_next = Gio.SimpleAction(name=self.ACTION_NEXT_NEW_IMG)
        self.action_next.connect('activate', self._next_new_img_activated_cb)

        self.action_choose = Gio.SimpleAction(name=self.ACTION_CHOOSE_LABEL)
        self.action_choose.connect('activate', self._choose_label_activated_cb)

    def do_activate(self):
        print("Info: quick label eog plugin activated")
        self._print_commands_summary()
        app = self.window.get_application()
        for k, v in self.ACTION_LABEL_SHORTCUT_MAP.items():
            self.window.add_action(self.action_label[k])
            app.set_accels_for_action( 'win.' + v, [k])

        self.window.add_action(self.action_next)
        app.set_accels_for_action( 'win.' + self.ACTION_NEXT_NEW_IMG, ['N'])

        self.window.add_action(self.action_choose)
        app.set_accels_for_action( 'win.' + self.ACTION_CHOOSE_LABEL, ['C'])

    def do_deactivate(self):
        print("Info: quick label eog plugin deactivated")
        for k, v in self.ACTION_LABEL_SHORTCUT_MAP.items():
            self.window.remove_action(v)

        self.window.remove_action(self.ACTION_NEXT_NEW_IMG)

        self.window.remove_action(self.ACTION_CHOOSE_LABEL)

    def _print_commands_summary(self):
        temp_map = self.ACTION_LABEL_SHORTCUT_MAP.copy()
        temp_map['N'] = self.ACTION_NEXT_NEW_IMG
        temp_map['C'] = self.ACTION_CHOOSE_LABEL
        print("Info: the shortcuts are", temp_map)

    def _debug(self, obj):
        for m in dir(obj):
            print(m, "-------------")#, getattr(obj, m))

    def _get_img_id(self, img):
        filename = os.path.basename(img.get_file().get_path())
        doc_id, file_id = re.sub(r'(_patch\d+_row\d+)?\.jpg','',filename).split('_')
        return int(doc_id), int(file_id)

    def _is_new_img(self, img):
        filename = os.path.basename(img.get_file().get_path())
        return self.label_df[self.label_df['filename'] == filename].empty

    def _change_img(self, next_img):
        view = self.window.get_thumb_view()
        if(next_img is not None): view.set_current_image(next_img, True)
        else:
            print("Info: last picture reached, exiting")
            self.label_df.to_csv(index=False, path_or_buf=self.csv_file)
            self.window.close()

    def _load_data(self, img):
        if(self.data_loaded): return
        dirpath = os.path.dirname(img.get_file().get_path())
        if(self.csv_file is None):
            self.csv_file = os.path.join(dirpath, self.LABEL_FILENAME)
            self.window.get_titlebar().set_subtitle("Target: " + self.csv_file)
        if(self.label_df is None):
            self.label_df = pd.read_csv(self.csv_file) if os.path.isfile(self.csv_file) else pd.DataFrame(
                columns=['filename', 'document_aim_id', 'file_aim_id', 'old_label', 'validation_label'])
            #load validated labels from master csv
            if(os.path.isfile(self.MASTER_VALIDATED_LABEL_CSV)):
                master_df = pd.read_csv(self.MASTER_VALIDATED_LABEL_CSV)
                #doesn't match patches image filenames
                files = [os.path.splitext(file)[0].split('_') for file in os.listdir(dirpath) if re.match(r'\d+_\d+\.jpg', file)]
                f_df = pd.DataFrame(files, columns=['document_aim_id', 'file_aim_id']).astype(int)
                master_df = pd.merge(master_df, f_df, how='inner', on=['document_aim_id', 'file_aim_id'])
                self.label_df = master_df.append(self.label_df, ignore_index=True, sort=False)

            self.label_df.drop_duplicates(subset=['filename'], keep='last', inplace=True)
            self.label_choices.update(self.label_df['validation_label'].unique().tolist())
        self.data_loaded = True

    def _next_new_img_activated_cb(self, action, param):
        img = self.window.get_image()
        if(img is None): return
        self._load_data(img)
        if(self._is_new_img(img)):
            print("Info: already at new unvalidated picture")
            return

        store = self.window.get_store()
        old_pos = pos = store.get_pos_by_image(img)
        is_new = False
        while not is_new:
            pos += 1
            next_img = store.get_image_by_pos(pos)
            if(next_img is None): break
            is_new = self._is_new_img(next_img)
        if(next_img is None): print("Info: fast forwarding ... all pictures are already labeled")
        else: print("Info: fast forwarding from validated picture", old_pos+1, "to next unvalidated picture", pos+1)
        self._change_img(next_img)

    def _choose_label_activated_cb(self, action, param):
        print("Choose label among", self.label_choices)
        msg ="Choose label in the list.\nTo create a new label, press cancel."
        title = "Choose label"
        choice = choicebox(msg, title, list(self.label_choices))
        if(choice is None):
            choice = enterbox("Enter new label:")
            if(choice is None):
                print("Warning: No new label provided")
                return
            else:
                self.label_choices.update([choice])
        self._writer(choice)

    def _writer(self, label):
        img = self.window.get_image()
        if(not img): return

        filepath = img.get_file().get_path()
        filename = os.path.basename(filepath)
        dirpath = os.path.dirname(filepath)
        old_label = os.path.basename(dirpath)
        self._load_data(img)

        temp = self.label_df.shape
        doc_id, file_id = self._get_img_id(img)
        self.label_df = self.label_df.append({'filename': filename, 'document_aim_id': doc_id, 'file_aim_id': file_id,
            'old_label': old_label, 'validation_label': label}, ignore_index=True)
        self.label_df.drop_duplicates(subset=['filename'], keep='last', inplace=True)
        update = (temp == self.label_df.shape)

        store = self.window.get_store()
        old_pos = store.get_pos_by_image(img)

        print("Info:", str(old_pos + 1) + "/" + str(store.length()), "update:" if update else "new:", filename, "with old label", old_label, "->", label)
        self.label_df.to_csv(index=False, path_or_buf=self.csv_file)

        if((old_pos + 1) % 10 ==0): self._print_commands_summary()

        next_img = store.get_image_by_pos(old_pos+1)
        self._change_img(next_img)

