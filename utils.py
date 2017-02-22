from colorama import init
from colorama import Fore, Back, Style


def print_module(string):

    init()
    print("\n" + Back.WHITE + Fore.BLACK + string + Style.RESET_ALL + "\n")


def print_dict(d):

    init()
    for k, v in d.iteritems():
        print(Style.BRIGHT + k + Style.RESET_ALL + ": %s" % v + Style.RESET_ALL)


def print_result(r):

    init()
    print(Style.BRIGHT + "Mean Time per update: " + Fore.GREEN + Back.WHITE + "%s" % r + Style.RESET_ALL)