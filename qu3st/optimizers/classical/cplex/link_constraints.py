def set_link_constraints(instance, model, t_vars):
    lf = instance.links_first()
    ls = instance.links_second()
    for l in range(len(lf)):
        model.add_constraint(t_vars[ls[l]] - t_vars[lf[l]] <= 0)
