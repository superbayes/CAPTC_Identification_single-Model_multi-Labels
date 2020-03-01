from generate_captcha import generateCaptcha

if __name__ == '__main__':
    g = generateCaptcha()
    for i in range(500):
        i = i+1
        X, Y = g.gen_test_captcha()
        print(i,X.shape,Y.shape)
