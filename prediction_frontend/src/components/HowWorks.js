import React, { Component } from 'react';

export default class HowWorks extends Component {
    render() {
        return (
            <div>
                <h1>How exactly are predictions made?</h1>
                <h2>1. A data collecting technique is used to gather data from the URL you specify</h2>
                <h2>2. The data is processed in accordance with a data science model that has been pre-generated</h2>
                <h2>3. This model, currently an LGBM classifier, predicts an outcome from the data.</h2>
                <h2>4. SHAP values are also recorded to demonstrate feature importance in the prediction made.</h2>
                <h2>5. The data is sent back here and displayed to you here :)</h2>
                <h2>Click <a href="https://www.kaggle.com/rafayk7/kickstarter-real">here</a> for the link to the training process for the model!</h2>
                <h2>The github repo is available <a href="https://github.com/rafayk7/predictstart">here</a>.</h2>
            </div>

        )
    }

}
