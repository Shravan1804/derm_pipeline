#! /usr/bin/env python2

#README:
#this script needs
#1.  python 2.7 (or at least < 3) https://www.python.org/downloads/release/python-278/
#        also python-tk and python-imaging-tk
#2.  imagemagick http://www.imagemagick.org/script/binary-releases.php#windows
#3.  both added to PATH http://stackoverflow.com/questions/6318156/adding-python-path-on-windows-7

#4. If "import Image" fails below, do this...
#    install pip http://stackoverflow.com/questions/4750806/how-to-install-pip-on-windows
#    run "pip install Pillow"
#    or on linux install python-pillow and python-pillow-tk http://stackoverflow.com/questions/10630736/no-module-named-image-tk

#you may change the below self-explanatory variables

#select input images from current directory
image_extensions = [".jpg", ".png", ".bmp"]

#directory to put output images (created automatically in current directory)
out_directory = "crops"

#after cropping, will resize down until the image firs in these dimensions. set to 0 to disable
resize_width = 0
resize_height = 0

#uses low resolution to show crop (real image will look better than preview)
fast_preview = True

#if the above is False, this controls how accurate the left hand preview image is
antialiase_original_preview = True

#ignores check to see if maintaining the apsect ratio perfectly is possible
allow_fractional_size = False
deskew_before_crop = True

def pause():
	raw_input("Press Enter to continue...")

def file_exists(filename):
	return os.path.exists(filename)

try:
	import os, sys
	print sys.version

	from distutils import spawn
	convert_path = spawn.find_executable("convert")
	if convert_path:
		print "Found 'convert' at", convert_path
	else:
		raise EnvironmentError("Could not find ImageMagick's 'convert'. Is it installed and in PATH?")

	print "Importing libraries..."

	print "> Tkinter"
	import Tkinter
	from Tkinter import *
	from Tkinter import Frame
	print "> subprocess"
	import subprocess
	print "> tkFileDialog"
	import tkFileDialog
	print "> shutil"
	import shutil
	print "> Image"
	try:
		from PIL import Image
	except ImportError:
		import Image
	print "> ImageOps"
	try:
		from PIL import ImageOps
	except ImportError:
		import ImageOps
	print "> ImageTk"
	try:
		from PIL import ImageTk
	except ImportError:
		import ImageTk
	print "Done"
except Exception as e:
	#because windows closes the window
	print e
	pause()
	raise e

if resize_height == 0:
	resize_width = 0
if resize_width < -1 or resize_height < -1:
	print "Note: resize is invalid. Not resizing."
	pause()
	resize_width = 0

class MyApp(Tk):
	def getImages(self, dir):
		print "Scanning " + dir
		allImages = []
		for i in os.listdir(dir):
			b, e = os.path.splitext(i)
			if e.lower() not in image_extensions: continue
			allImages += [i]
		allImages.sort()
		return allImages

	def __init__(self):
		Tk.__init__(self)

		self.inDir = os.getcwd()

		infiles = self.getImages(self.inDir)

		if not len(infiles):
			print "No images in the current directory. Please select a different directory."
			self.inDir = tkFileDialog.askdirectory(parent=self, initialdir="/",title='Please select a directory')
			if not len(self.inDir):
				print "No directory selected. Exiting."
				pause()
				raise SystemExit()
			self.inDir = os.path.normpath(self.inDir)
			infiles = self.getImages(self.inDir)
			if not len(infiles):
				print "No images found in " + self.inDir + ". Exiting."
				pause()
				raise SystemExit()
			print "Found", len(infiles), "images"
		else:
			print "Found", len(infiles), "images in the current directory"

		self.outDir = os.path.join(self.inDir, out_directory)

		if not os.path.exists(self.outDir):
			print "Creating output directory, " + self.outDir
			os.makedirs(self.outDir)

		print "Initializing GUI"

		self.grid_rowconfigure(0, weight=1)
		self.grid_columnconfigure(0, weight=1)
		self.geometry("1024x512")
		#self.resizable(0,0)

		self.files = infiles

		self.preview = None

		self.item = None
		self.cropIndex = 2
		self.x = 0
		self.y = 0
		self.current = 0
		self.cropdiv = 1.06
		self.previewBoxWidget = None
		self.originalBoxWidget = None
		self.canvas = None

		self.controls = Frame(self)
		self.controls.grid(row=1, column=0, columnspan=2, sticky="nsew")
		self.buttons = []
		self.info = Label(self.controls, text="Pyar's Cropper")
		self.info.grid(row=0, column=0, sticky="nsew")

		self.inputs = []
		self.aspect = (StringVar(), StringVar())
		self.aspect[0].set("8.5")
		self.aspect[1].set("11")
		self.inputs += [Entry(self.controls, textvariable=self.aspect[0])]
		self.inputs[-1].grid(row=0, column=1, sticky="nsew")
		self.inputs += [Entry(self.controls, textvariable=self.aspect[1])]
		self.inputs[-1].grid(row=0, column=2, sticky="nsew")

		self.buttons += [Button(self.controls, text="Prev", command=self.previous)]
		self.buttons[-1].grid(row=0, column=3, sticky="nsew")
		self.buttons += [Button(self.controls, text="Next", command=self.next)]
		self.buttons[-1].grid(row=0, column=4, sticky="nsew")
		self.buttons += [Button(self.controls, text="RotateL", command=self.rotate_l)]
		self.buttons[-1].grid(row=0, column=5, sticky="nsew")
		self.buttons += [Button(self.controls, text="RotateR", command=self.rotate_r)]
		self.buttons[-1].grid(row=0, column=6, sticky="nsew")
		self.buttons += [Button(self.controls, text="Copy", command=self.copy)]
		self.buttons[-1].grid(row=0, column=7, sticky="nsew")
		self.buttons += [Button(self.controls, text="Resize", command=self.resize)]
		self.buttons[-1].grid(row=0, column=8, sticky="nsew")
		self.buttons += [Button(self.controls, text="Crop", command=self.save_next)]
		self.buttons[-1].grid(row=0, column=9, sticky="nsew")

		self.restrictSizes = IntVar()
		self.inputs += [Checkbutton(self.controls, text="Perfect Pixel Ratio", variable=self.restrictSizes)]
		self.inputs[-1].grid(row=0, column=10, sticky="nsew")

		self.imageLabel = Canvas(self, highlightthickness=0)
		self.imageLabel.grid(row=0, column=0, sticky='nw', padx=0, pady=0)
		self.c = self.imageLabel

		self.previewLabel = Label(self, relief=FLAT, borderwidth=0)
		self.previewLabel.grid(row=0, column=1, sticky='nw', padx=0, pady=0)

		self.restrictSizes.set(0 if allow_fractional_size else 1)

		self.aspect[0].trace("w", self.on_aspect_changed)
		self.aspect[1].trace("w", self.on_aspect_changed)
		self.restrictSizes.trace("w", self.on_option_changed)
		self.bind('q', self.previous)
		self.bind('w', self.next)
		self.bind('a', self.rotate_l)
		self.bind('s', self.rotate_r)
		self.bind('<BackSpace>', self.backspace_key)
		self.bind('<space>', self.save_next)
		self.bind('<Up>', self.up_key)
		self.bind('<Left>', self.left_key)
		self.bind('<Right>', self.right_key)
		self.bind('<Down>', self.down_key)
		self.bind('<Control-Up>', self.control_up_key)
		self.bind('<Control-Left>', self.control_left_key)
		self.bind('<Control-Right>', self.control_right_key)
		self.bind('<Control-Down>', self.control_down_key)
		self.c.bind('<ButtonPress-1>', self.on_mouse_down)
		self.c.bind('<B1-Motion>', self.on_mouse_drag)
		self.c.bind('<ButtonRelease-1>', self.on_mouse_up)
		#self.c.bind('<Button-3>', self.on_right_click)
		self.bind('<Button-4>', self.on_mouse_scroll)
		self.bind('<Button-5>', self.on_mouse_scroll)
		self.bind('<MouseWheel>', self.on_mouse_scroll)

		self.start_cropping()

	def current_filename(self):
		return self.files[self.current]

	def current_full_filepath(self):
		return os.path.join(self.inDir, self.current_filename())

	def output_full_filepath(self, index=None):
		if index is not None:
			return os.path.join(self.outDir, self.files[index])
		else:
			return os.path.join(self.outDir, self.files[self.current])

	def deskewed_filename(self):
		return "deskewed_" + self.current_filename()

	def deskewed_full_filepath(self):
		return os.path.join(self.outDir, self.deskewed_filename())

	def output_already_exists(self):
		exists = file_exists(self.output_full_filepath())
		if exists:
			print self.output_full_filepath() + " already exists"

		return exists

	def deskewed_already_exists(self):
		exists = file_exists(self.deskewed_full_filepath())
		if exists:
			print self.deskewed_full_filepath() + " already exists"

		return exists

	def start_cropping(self):
		print "Checking for existing crops"

		while self.current < len(self.files) and self.output_already_exists():
			print "Skipping " + self.files[self.current] + ". Already cropped."
			self.next(False)

		self.load_imgfile()

	def updateCropSize(self):
		return True
		# if self.cropIndex <= 4:
			#self.cropdiv = 8.0 / (9.0 - self.cropIndex)
		# else:
			#self.cropdiv = (1 + (self.cropIndex - 1) * 0.25)

	def getCropSize(self):
		self.updateCropSize()

		if self.imageOrigSize[0] > self.imageOrigSize[1]: # wider than taller
			h = int(self.imageOrigSize[1] / self.cropdiv)
			w = int(self.imageOrigSize[1] * self.aspectRatio / self.cropdiv)
		else:
			w = int(self.imageOrigSize[0] / self.cropdiv)
			h = int(self.imageOrigSize[0] / (self.cropdiv * self.aspectRatio))

		#w = int(self.imageOrigSize[0] / self.cropdiv)

		return w, h

	def getRealBox(self):
		w, h = self.getCropSize()
		imw = self.imageOrigSize[0]
		imh = self.imageOrigSize[1]
		prevw = self.imagePhoto.width()
		prevh = self.imagePhoto.height()
		# box = (int(round(self.x*imw/prevw))-w//2, int(round(self.y*imh/prevh))-h//2)
		box = (int(round(self.x*imw/prevw))-w, int(round(self.y*imh/prevh)))
		box = (max(box[0], 0), max(box[1], 0))
		box = (min(box[0]+w, imw)-w, min(box[1]+h, imh)-h)
		box = (box[0], box[1], box[0]+w, box[1]+h)
		# print "top: " + str(box[0]) + ", left: " + str(box[1]) + ", width: " + str(w) + ", height: " + str(h)
		return box

	def getPreviewBox(self):
		imw = self.imageOrigSize[0]
		imh = self.imageOrigSize[1]
		prevw = self.imagePhoto.width()
		prevh = self.imagePhoto.height()
		bbox = self.getRealBox()
		bbox = (bbox[0]*prevw/imw, bbox[1]*prevh/imh, bbox[2]*prevw/imw, bbox[3]*prevh/imh)
		return (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))

	def previous(self, event=None):
		self.current -= 1
		self.current = (self.current + len(self.files)) % len(self.files)
		self.load_imgfile()

	def next(self, load=True):
		self.current += 1
		self.current = (self.current + len(self.files)) % len(self.files)
		if load == True:
			self.load_imgfile()

	def copy(self):
		c = "copy \"" + self.current_full_filepath() + "\" \"" + self.output_full_filepath() + "\""
		print c
		shutil.copy(self.current_full_filepath(), self.output_full_filepath())
		self.next()

	def resize(self):
		if not (resize_width > 0):
			print "Error: no resize specified. Not resizing"
			return
		c = "convert \"" + self.current_full_filepath() + "\""
		c += " -resize \"" + str(resize_width) + "x" + str(resize_height) + ">\""
		c += " \"" + self.output_full_filepath() + "\""
		print c
		subprocess.Popen(c, shell=True)
		print "Running"
		self.next()

	def rotate_l(self, event=None):
		self.rotate()

	def rotate_r(self, event=None):
		self.rotate("R")

	def rotate(self, direction="L"):
		if direction == "R":
			angle = "90"
		else:
			angle = "270"

		command = 'jpegtran -rotate '+angle+' -trim -outfile "' + self.current_full_filepath() + '" "' + self.current_full_filepath() + '"'
		print command
		subprocess.call(command, shell=True)
		self.load_imgfile()

	def save_next(self, event=None):
		box = self.getRealBox()
		c = "convert \"" + self.current_full_filepath() + "\""
		c += " -crop " + str(box[2]-box[0]) + "x" + str(box[3]-box[1]) + "+" + str(box[0]) + "+" + str(box[1])
		if (resize_width > 0):
			c += " -resize \"" + str(resize_width) + "x" + str(resize_height) + ">\""
		c += " \"" + self.output_full_filepath() + "\""
		print "Running: " + c
		subprocess.Popen(c, shell=True)
		# self.deskew(False)
		self.next()

	def deskew(self, load = True):
		command = "convert \"" + self.output_full_filepath()+ "\""
		command += " -background black -deskew 40% -trim "
		command += "\"" + self.deskewed_full_filepath() + "\""
		print command
		subprocess.Popen(command, shell=True)
		if(load):
			self.load_imgfile()

	def load_imgfile(self):
		filename          = self.current_filename()
		fullFilename      = self.current_full_filepath()
		outputFilename    = self.output_full_filepath()

		if self.output_already_exists():
			return

		# if not self.deskewed_already_exists() and deskew_before_crop:
		# 	self.deskew(False)
		# 	fullFilename = self.deskewed_full_filepath()

		print "Loading " + fullFilename
		img = Image.open(fullFilename)

		self.imageOrig = img
		self.imageOrigSize = (img.size[0], img.size[1])
		print "Image is " + str(self.imageOrigSize[0]) + "x" + str(self.imageOrigSize[1])

		basewidth = 512
		wpercent = (basewidth/float(img.size[0]))
		hsize = int((float(img.size[1])*float(wpercent)))
		if fast_preview:
			#does NOT create a copy so self.imageOrig is the same as self.image
			img.thumbnail((basewidth,hsize), Image.NEAREST)
		else:
			if antialiase_original_preview:
				img = img.resize((basewidth,hsize), Image.ANTIALIAS)
			else:
				img = img.copy()
				img.thumbnail((basewidth,hsize), Image.NEAREST)

		self.image = img
		print "Resized preview"

		#self.geometry("1024x"+str(hsize + 100))
		self.configure(relief='flat', background='red')

		self.imagePhoto = ImageTk.PhotoImage(self.image)
		self.imageLabel.configure(width=self.imagePhoto.width(), height=self.imagePhoto.height())
		self.imageLabel.create_image(0, 0, anchor=NW, image=self.imagePhoto)

		self.previewPhoto = ImageTk.PhotoImage(self.image)
		self.previewLabel.configure(image=self.previewPhoto)

		self.item = None

		self.on_aspect_changed(None, None, None) #update aspect ratio with new image size

	def test(self):
		if not allow_fractional_size:
			self.updateCropSize()
			if int(self.cropdiv) != self.cropdiv: return False
			w, h = self.getCropSize()
			if int(h) != h or int(w) != w: return False
		return True

	def update_box(self, widget):
		bbox = self.getPreviewBox()

		if self.item is None:
			self.item = widget.create_rectangle(bbox, outline="red")
		else:
			try:
				widget.coords(self.item, *bbox)
				self.canvas = widget
			except AttributeError:
				# this was updated via keyboard
				self.canvas.coords(self.item, *bbox)

	def update_preview(self, widget):
		if self.item:
			#get a crop for the preview
			#box = tuple((int(round(v)) for v in widget.coords(self.item)))
			box = self.getRealBox()
			pbox = self.getPreviewBox()
			if fast_preview:
				preview = self.image.crop(pbox) # region of interest
			else:
				preview = self.imageOrig.crop(box) # region of interest

			#add black borders for correct aspect ratio
			#if preview.size[0] > 512:
			preview.thumbnail(self.image.size, Image.ANTIALIAS) #downscale to preview rez
			paspect = preview.size[0]/float(preview.size[1])
			aspect = self.image.size[0]/float(self.image.size[1])
			if paspect < aspect:
				bbox = (0, 0, int(preview.size[1] * aspect), preview.size[1])
			else:
				bbox = (0, 0, preview.size[0], int(preview.size[0] / aspect))
			preview = ImageOps.expand(preview, border=(bbox[2]-preview.size[0], bbox[3]-preview.size[1]))
			#preview = ImageOps.fit(preview, size=self.image.size, method=Image.ANTIALIAS, bleed=-10.0)

			#resize to preview rez (if too small)
			self.preview = preview.resize(self.image.size, Image.ANTIALIAS)
			self.previewPhoto = ImageTk.PhotoImage(self.preview)
			self.previewLabel.configure(image=self.previewPhoto)

			print str(box[2]-box[0])+"x"+str(box[3]-box[1])+"+"+str(box[0])+"+"+str(box[1])

	def on_aspect_changed(self, event, var1, var2):
		try:
			x = float(self.aspect[0].get())
			y = float(self.aspect[1].get())
			if x < 0 or y < y:
				raise ZeroDivisionError()
			self.aspectRatio = x / y
		except:
			self.aspectRatio = float(self.imageOrigSize[0])/float(self.imageOrigSize[1])
		self.update_box(self.imageLabel)
		self.update_preview(self.imageLabel)

	def on_option_changed(self, event, var1, var2):
		global allow_fractional_size
		allow_fractional_size = (self.restrictSizes.get() == 0)

	def on_mouse_scroll(self, event):
		SCROLL_UP = 1
		SCROLL_DOWN = -1

		if event.num == 5 or event.delta < 0:
			dir = SCROLL_DOWN
		if event.num == 4 or event.delta > 0:
			dir = SCROLL_UP

		if dir == SCROLL_UP:
			self.cropdiv += 0.01
			if self.cropdiv > 3.0:
				self.cropdiv = 3.0

		if dir == SCROLL_DOWN:
			self.cropdiv -= 0.01
			if self.cropdiv < 0.1:
				self.cropdiv = 0.1

		print str(self.cropdiv)
		self.update_box(self.imageLabel)
		self.update_preview(self.imageLabel)

	def on_mouse_down(self, event):
		self.x = event.x
		self.y = event.y
		self.update_box(event.widget)

	def up_key(self, event):
		self.y -= 1
		self.update_box(event.widget)
		self.update_preview(event.widget)

	def left_key(self, event):
		self.x -= 1
		self.update_box(event.widget)
		self.update_preview(event.widget)

	def down_key(self, event):
		self.y += 1
		self.update_box(event.widget)
		self.update_preview(event.widget)

	def right_key(self, event):
		self.x += 1
		self.update_box(event.widget)
		self.update_preview(event.widget)

	def on_mouse_drag(self, event):
		self.x = event.x
		self.y = event.y
		self.update_box(event.widget)

	def on_mouse_up(self, event):
		self.update_preview(event.widget)

	def control_up_key(self, event):
		self.y -= 5
		self.update_box(event.widget)
		self.update_preview(event.widget)

	def control_left_key(self, event):
		self.x -= 5
		self.update_box(event.widget)
		self.update_preview(event.widget)

	def control_down_key(self, event):
		self.y += 5
		self.update_box(event.widget)
		self.update_preview(event.widget)

	def control_right_key(self, event):
		self.x += 5
		self.update_box(event.widget)
		self.update_preview(event.widget)

	def backspace_key(self, event):
		self.previous()

app =  MyApp()
app.mainloop()
