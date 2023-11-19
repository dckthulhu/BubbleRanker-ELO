#!/usr/bin/env python
"""
Matplotlib based photo ranking system using the Elo rating system.

Reference: http://en.wikipedia.org/wiki/Elo_rating_system

by Nick Hilton

This file is in the public domain.

"""

# Python
import argparse
import glob
import json
import os
import sys


# 3rd party
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import exifread

directory_path = "G:\AI\ELO SORT\Groups"
filelist = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]



class Photo:

    LEFT = 0
    RIGHT = 1

    def __init__(self, filename, score = 1400.0, wins = 0, matches = 0):

        if not os.path.isfile(filename):
            raise ValueError("Could not find the file: %s" % filename)

        self._filename = filename
        self._score = score
        self._wins = wins
        self._matches = matches

        self._read_and_downsample()


    def data(self):
        return self._data


    def filename(self):
        return self._filename


    def matches(self):
        return self._matches


    def score(self, s = None, is_winner = None):

        if s is None:
            return self._score

        assert is_winner is not None

        self._score = s

        self._matches += 1

        if is_winner:
            self._wins += 1


    def win_percentage(self):
        return 100.0 * float(self._wins) / float(self._matches)


    def __eq__(self, rhs):
        return self._filename == rhs._filename


    def to_dict(self):

        return {
            'filename' : self._filename,
            'score' : self._score,
            'matches' : self._matches,
            'wins' : self._wins,
        }


    def _read_and_downsample(self):
        """
        Reads the image, performs rotation, and downsamples.
        """

        #----------------------------------------------------------------------
        # read image

        f = self._filename

        data = mpimg.imread(f)

        #----------------------------------------------------------------------
        # downsample

        # the point of downsampling is so the images can be redrawn by the
        # display as fast as possible, this is so one can iterate though the
        # image set as quickly as possible.  No one want's to wait around for
        # the fat images to be loaded over and over.

        # dump downsample, just discard columns-n-rows

        print("Before downsampling - data shape:", data.shape)

        M, N = data.shape[0:2]

        MN = max([M, N])

        step = int(MN / 800)
        if step == 0:
            step = 1

        if data.ndim == 2:
        # Add a third dimension (assume grayscale image)
            data = data[:, :, np.newaxis]

            data = data[0:M:step, 0:N:step, :]

        print("After downsampling - data shape:", data.shape)


        #----------------------------------------------------------------------
        # rotate

        # read orientation with exifread

        with open(f, 'rb') as fd:
            tags = exifread.process_file(fd)

        r = 'Horizontal (normal)'

        try:
            orientation_tag = tags.get('Image Orientation')
            if orientation_tag:
                r = str(orientation_tag)

        except:
            pass

        # rotate as necessary

        if r == 'Horizontal (normal)' or r == '0':
            pass

        elif r == 'Rotated 90 CW':

            data = np.rot90(data, 3)

        elif r == 'Rotated 90 CCW':

            data = np.rot90(data, 1)

        elif r == 'Rotated 180':

            data = np.rot90(data, 2)

        else:
            print(f"Unhandled rotation: {r}")
            raise RuntimeError('Unhandled rotation "%s"' % r)

        self._data = data


class SetPhoto:
    # Similar to the Photo class, but represents a set of photos with a common prefix.

    def __init__(self, username, photos=None):
        self._username = username
        self._photos = photos or []  # List to store individual photos in the set
        self._score = 1400.0  # You can initialize the score as needed
        self._wins = 0
        self._matches = 0

    def add_photo(self, photo):
        self._photos.append(photo)

    # ... (Methods for getting username, photos, score, etc.)

class SetEloTable:
    # Similar to EloTable, but for sets of photos.

    def SomeInitialization(self):
    # Your initialization code here
        set_table = SetEloTable()
        return set_table
        
    
    def __init__(self, max_increase=32.0):
        self._K = max_increase
        self._set_photos = {}
        self._shuffled_keys = []

    def add_set_photo(self, username, photo):
        if username not in self._set_photos:
            self._set_photos[username] = SetPhoto(username)
        self._set_photos[username].add_photo(photo)


class SetDisplay:
    # Similar to Display, but for sets of photos.

    def __init__(self, set_photo1, set_photo2, title=None, figsize=None):
        # Display the set photos instead of individual photos

    # ... (Similar changes in other methods)

        def main():

            ranking_table_json = 'set_ranking_table.json'
            ranked_txt = 'set_ranked.txt'

    # Create the ranking table for sets
            set_table = SetEloTable().SomeInitialization()

            for f in filelist:
        # Assume file names are in the format "username_01.jpg", "username_02.jpg", etc.
                username = f.split('_')[0]
                set_table.add_set_photo(username, f)

    # Rank the sets of photos
                set_table.rank_set_photos(args.n_rounds, args.figsize)

class Display(object):
    """
    Given two photos, displays them with Matplotlib and provides a graphical
    means of choosing the better photo.

    Click on the select button to pick the better photo.

    ~OR~

    Press the left or right arrow key to pick the better photo.

    """


    def __init__(self, f1, f2, title = None, figsize = None):

        self._choice = None

        assert isinstance(f1, Photo)
        assert isinstance(f2, Photo)

        if figsize is None:
            figsize = [20,12]

        fig = plt.figure(figsize=figsize)

        # Get the screen width and height
        screen_width, screen_height = plt.gcf().canvas.get_width_height()

        # Calculate the center position
        center_x = (screen_width - figsize[0]) / 2
        center_y = (screen_height - figsize[1]) / 2

        # Set the window position to the center
        fig.canvas.manager.window.wm_geometry(f"+{int(center_x)}+{int(center_y)}")


        h = 10

        ax11 = plt.subplot2grid((h,2), (0,0), rowspan = h - 1)
        ax12 = plt.subplot2grid((h,2), (0,1), rowspan = h - 1)

        ax21 = plt.subplot2grid((h,6), (h - 1, 1))
        ax22 = plt.subplot2grid((h,6), (h - 1, 4))

        kwargs = dict(s = 'Select', ha = 'center', va = 'center', fontsize=20)

        ax21.text(0.5, 0.5, **kwargs)
        ax22.text(0.5, 0.5, **kwargs)

        self._fig = fig
        self._ax_select_left = ax21
        self._ax_select_right = ax22

        fig.subplots_adjust(
            left = 0.02,
            bottom = 0.02,
            right = 0.98,
            top = 0.98,
            wspace = 0.05,
            hspace = 0,
        )

        ax11.imshow(f1.data())
        ax12.imshow(f2.data())

        for ax in [ax11, ax12, ax21, ax22]:
            ax.set_xticklabels([])
            ax.set_yticklabels([])

            ax.set_xticks([])
            ax.set_yticks([])

        self._attach_callbacks()

        if title:
            fig.suptitle(title, fontsize=20)

        plt.show()


    def _on_click(self, event):

        if event.inaxes == self._ax_select_left:
            self._choice = Photo.LEFT
            plt.close(self._fig)

        elif event.inaxes == self._ax_select_right:
            self._choice = Photo.RIGHT
            plt.close(self._fig)


    def _on_key_press(self, event):

        if event.key == 'left':
            self._choice = Photo.LEFT
            plt.close(self._fig)

        elif event.key == 'right':
            self._choice = Photo.RIGHT
            plt.close(self._fig)


    def _attach_callbacks(self):
        self._fig.canvas.mpl_connect('button_press_event', self._on_click)
        self._fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        # Center the window on the screen
        manager = plt.get_current_fig_manager()
        screen_width, screen_height = manager.window.wm_maxsize()
        center_x = (screen_width - self._fig.get_figwidth()) / 2
        center_y = (screen_height - self._fig.get_figheight()) / 2
        manager.window.wm_geometry("+1400+100")





class EloTable:


    def __init__(self, max_increase = 32.0):
        self._K = max_increase
        self._photos = {}
        self._shuffled_keys = []


    def add_photo(self, filename_or_photo):

        if isinstance(filename_or_photo, str):

            filename = filename_or_photo

            if filename not in self._photos:
                self._photos[filename] = Photo(filename)

        elif isinstance(filename_or_photo, Photo):

            photo = filename_or_photo

            if photo.filename() not in self._photos:
                self._photos[photo.filename()] = photo


    def get_ranked_list(self):

        # Convert the dictionary into a list and then sort by score.

        ranked_list = self._photos.values()

        ranked_list = sorted(
            ranked_list,
            key = lambda record : record.score(),
            reverse = True)

        return ranked_list


    def rank_photos(self, n_iterations, figsize):
        """
        Displays two photos using the command "gnome-open".  Then asks which
        photo is better.
        """

        n_photos = len(self._photos)

        keys = list(self._photos.keys())

        for i in range(n_iterations):

            np.random.shuffle(keys)

            n_matchups = n_photos / 2

            for j in range(0, n_photos - 1, 2):

                match_up = j / 2

                title = 'Round %d / %d, Match Up %d / %d' % (
                    i + 1, n_iterations,
                    match_up + 1,
                    n_matchups)

                photo_a = self._photos[keys[j]]
                photo_b = self._photos[keys[j+1]]

                d = Display(photo_a, photo_b, title, figsize)

                if d._choice == Photo.LEFT:
                    self.__score_result(photo_a, photo_b)
                elif d._choice == Photo.RIGHT:
                    self.__score_result(photo_b, photo_a)
                else:
                    raise RuntimeError("oops, found a bug!")


    def __score_result(self, winning_photo, loosing_photo):

        # Current ratings
        R_a = winning_photo.score()
        R_b = loosing_photo.score()

        # Expectation

        E_a = 1.0 / (1.0 + 10.0 ** ((R_a - R_b) / 400.0))

        E_b = 1.0 / (1.0 + 10.0 ** ((R_b - R_a) / 400.0))

        # New ratings
        R_a = R_a + self._K * (1.0 - E_a)
        R_b = R_b + self._K * (0.0 - E_b)

        winning_photo.score(R_a, True)
        loosing_photo.score(R_b, False)


    def to_dict(self):

        rl = self.get_ranked_list()

        rl = [x.to_dict() for x in rl]

        return {'photos' : rl}



def main():

    description = """\
Uses the Elo ranking algorithm to sort your images by rank.  The program globs
for .jpg images to present to you in random order, then you select the better
photo.  After n-rounds, the results are reported.

Click on the "Select" button or press the LEFT or RIGHT arrow to pick the
better photo.

"""
    parser = argparse.ArgumentParser(description = description)

    parser.add_argument(
        "-r",
        "--n-rounds",
        type = int,
        default = 3,
        help = "Specifies the number of rounds to pass through the photo set (3)"
    )

    parser.add_argument(
        "-f",
        "--figsize",
        nargs = 2,
        type = int,
        default = [20, 12],
        help = "Specifies width and height of the Matplotlib figsize (20, 12)"
    )

    parser.add_argument(
        "photo_dir",
        help = "The photo directory to scan for .jpg images"
    )

    args = parser.parse_args()

    assert os.path.isdir(args.photo_dir)

    os.chdir(args.photo_dir)

    ranking_table_json = 'ranking_table.json'
    ranked_txt         = 'ranked.txt'

    # Create the ranking table and add photos to it.

    table = EloTable()

    #--------------------------------------------------------------------------
    # Read in table .json if present

    sys.stdout.write("Reading in photos and downsampling ...")
    sys.stdout.flush()

    if os.path.isfile(ranking_table_json):
        with open(ranking_table_json, 'r') as fd:
            d = json.load(fd)

        # read photos and add to table

        for p in d['photos']:

            photo = Photo(**p)

            table.add_photo(photo)

    #--------------------------------------------------------------------------
    # glob for files, to include newly added files

    filelist = glob.glob('*.jpg')

    for f in filelist:
        table.add_photo(f)

    print(" done!")

    #--------------------------------------------------------------------------
    # Rank the photos!

    table.rank_photos(args.n_rounds, args.figsize)

    #--------------------------------------------------------------------------
    # save the table

    with open(ranking_table_json, 'w') as fd:

        d = table.to_dict()

        jstr = json.dumps(d, indent = 4, separators=(',', ' : '))

        fd.write(jstr)

    #--------------------------------------------------------------------------
    # dump ranked list to disk

    with open(ranked_txt, 'w') as fd:

        ranked_list = table.get_ranked_list()

        heading_fmt = "%4d    %4.0f    %7d    %7.2f    %s\n"

        heading = "Rank    Score    Matches    Win %    Filename\n"

        fd.write(heading)

        for i, photo in enumerate(ranked_list):

            line = heading_fmt %(
                i + 1,
                photo.score(),
                photo.matches(),
                photo.win_percentage(),
                photo.filename())

            fd.write(line)

    #--------------------------------------------------------------------------
    # dump ranked list to screen

    print("Final Ranking:")

    with open(ranked_txt, 'r') as fd:
        text = fd.read()

    print(text)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("An error occurred:", e)
        input("Press Enter to exit...")
