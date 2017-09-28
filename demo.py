print('ddd')
print('dfafafafa')
print('fdafafa')
print('ffdafasf')

content = 'If you are the proxy administrator, please put the required file(s)in the (confdir)/templates directory. The location of the (confdir) directory is specified in the main Privoxy config file. '

import jieba

print(" ".join(jieba.cut(content)))
