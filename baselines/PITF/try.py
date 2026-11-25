def fit(self, data, validation=None):
    exp_item_coff = 0.1
    if self.data_shape is None:
        self.data_shape = self._calc_number_of_dimensions(data, validation)

    self.latent_vector_ = self._init_latent_vectors(self.data_shape)
    remained_iter = self.max_iter
    user2item_set = get_user2item(data)
    while True:
        print(remained_iter)
        loss = 0
        remained_iter -= 1
        np.random.shuffle(data)
        for u, i, t in data:
            for pt in t:
                ni, nt = self._draw_negative_sample(user2item_set, u, i, t)
                y_diff = self.y(u, i, pt) - self.y(u, i, nt)
                y_diff_item = self.y1(u, i) - self.y1(u, ni)
                loss += (np.log(1e-10 + self._sigmoid(y_diff)) +
                         exp_item_coff * np.log(1e-10 + self._sigmoid(y_diff_item)))
                delta = 1 - self._sigmoid(y_diff)
                delta_item = exp_item_coff * (1 - self._sigmoid(y_diff_item))
                self.latent_vector_['u'][u] += self.alpha * (
                        delta * (self.latent_vector_['tu'][pt] - self.latent_vector_['tu'][nt]) + delta_item *
                        (self.latent_vector_['i'][i] - self.latent_vector_['i'][ni]) - self.lamb *
                        self.latent_vector_['u'][u])
                self.latent_vector_['i'][i] += self.alpha * (
                        delta * (self.latent_vector_['ti'][pt] - self.latent_vector_['ti'][nt]) + delta_item *
                        self.latent_vector_['u'][u] - self.lamb * self.latent_vector_['i'][i])
                self.latent_vector_['i'][ni] += self.alpha * (
                        delta_item * -self.latent_vector_['u'][u] - self.lamb * self.latent_vector_['i'][ni])
                self.latent_vector_['tu'][pt] += self.alpha * (
                        delta * self.latent_vector_['u'][u] - self.lamb * self.latent_vector_['tu'][pt])
                self.latent_vector_['tu'][nt] += self.alpha * (
                        delta * -self.latent_vector_['u'][u] - self.lamb * self.latent_vector_['tu'][nt])
                self.latent_vector_['ti'][pt] += self.alpha * (
                        delta * self.latent_vector_['i'][i] - self.lamb * self.latent_vector_['ti'][pt])
                self.latent_vector_['ti'][nt] += self.alpha * (
                        delta * -self.latent_vector_['i'][i] - self.lamb * self.latent_vector_['ti'][nt])
        if self.verbose == 1:
            print("%s\t%s" % (self.max_iter - remained_iter, self._score(validation)))
        if remained_iter <= 0:
            break
        print(loss)
    return self