def context_and_keystrokes(text):
    if text.endswith(' '):
        keystrokes = ''
        context = text.strip()
    else:
        split = text.split()
        if len(split) > 0:
            keystrokes = text.split()[-1]
        else:
            keystrokes = ''
        context = text[:-len(keystrokes)]
    return context, keystrokes

