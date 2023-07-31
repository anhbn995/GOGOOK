def get_successful_payment(payment_method, amount, plan_name, order_no):
    return """\
        <!doctype html>
            <html lang="en">
            <head>
            <meta charset="UTF-8">
            <meta name="viewport"
                    content="width=device-width, user-scalable=no, initial-scale=1.0, maximum-scale=1.0, minimum-scale=1.0">
            <meta http-equiv="X-UA-Compatible" content="ie=edge">
            <link
                href="https://fonts.googleapis.com/css2?family=Roboto:wght@100;200;300;400;500&display=swap"
                rel="stylesheet">
            <style>
                *,
                *::before,
                *::after {
                box-sizing: inherit;
                }

                a {
                text-decoration: none;
                }

                body {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
                color: #000;
                font-family: 'Roboto', sans-serif;
                }

                .container {
                padding-right: 15px;
                padding-left: 15px;
                margin-right: auto;
                margin-left: auto;
                }

                @media (min-width: 768px) {
                .container {
                    width: 600px;
                }
                }

                @media (min-width: 992px) {
                .container {
                    width: 900px;
                }
                }

                @media (min-width: 1200px) {
                .container {
                    width: 1100px;
                }
                }

                .header {
                background-color: #222266;
                color: #fff;
                }

                .top-header {
                padding-top: 16px;
                padding-bottom: 12px;
                }

                .bottom-header {
                padding-top: 8px;
                padding-bottom: 24px;
                margin-bottom: 24px;
                }

                .bottom-header__thumb {
                text-align: center;
                }

                .bottom-header__img {
                display: inline-block;
                border-radius: 50%;
                background-color: #d9e0f1;
                }

                .bottom-header__img img {
                margin: 24px;
                }

                .bottom-header__title {
                margin-top: 8px;
                text-align: center;
                font-size: 24px;
                }

                .bottom-header__subtitle {
                margin-top: 12px;
                text-align: center;
                font-weight: 200;
                font-size: 15px;
                }

                .main {
                padding-top: 24px;
                padding-bottom: 24px;
                }

                .main__line {
                margin: 20px 0;
                border-top: 1px solid rgba(0, 0, 0, 0.1);
                }

                .headline__title {
                font-weight: 500;
                }

                .headline__content {
                margin-top: 8px;
                font-weight: 300;
                font-size: 15px;
                }

                .order {
                margin-top: 16px;
                }

                .order__title {
                font-weight: 500;
                }

                .order__content {
                display: flex;
                }

                .order__plan {
                margin-top: 8px;
                font-size: 15px;
                font-weight: 400;
                color: #222266;
                }

                .order__caption {
                font-size: 14px;
                font-weight: 300;
                }

                .order__price {
                margin-top: 8px;
                font-weight: 500;
                margin-left: auto;
                color: #dc6618;
                }

                .payment {
                margin-top: 16px;
                }

                .payment__name {
                font-weight: 500;
                }

                .payment__content {
                margin-top: 8px;
                font-weight: 300;
                font-size: 15px;
                }

                .payment__caption {
                font-size: 14px;
                }

                .thank__title {
                font-weight: 500;
                }

                .thank__content {
                margin-top: 4px;
                font-weight: 300;
                font-size: 15px;
                }

                .footer {
                padding-top: 8px;
                padding-bottom: 32px;
                font-weight: 300;
                font-size: 15px;
                }

                .footer__link {
                font-weight: 500;
                color: #222266;
                }
            </style>
            </head>
            <body>
            <div class="header">
            <div class="container">
                <div class="top-header">
                <img height="48px" src="https://api.eofactory.ai/file/storage/1586916503761_logo-home.png?fbclid=IwAR39VGX034m_wiNnCTZmLk7kYje46An5Ex8B4vXASBbw28iK7IeX3YsOYWk" alt="logo">
                </div>
                <div class="bottom-header">
                <div class="bottom-header__thumb">
                    <div class="bottom-header__img">
                    <img height="80px" src="https://api.eofactory.ai/file/storage/1586917144354_cart.png?fbclid=IwAR0hwD0xlQ8FiFKsEEHPUnBgwkPfCrweOKV1NPFXUyyt_fXeMqTVO2a-5sE" alt="cart">
                    </div>
                </div>
                <div class="bottom-header__title">
                    <span>Order recevided</span>
                </div>
                <div class="bottom-header__subtitle">
                    <span>Order no: """ +order_no+ """</span>
                </div>
                </div>
            </div>
            </div>
            <div class="main">
            <div class="container">
                <div class="headline">
                <div class="headline__title">Hi User, we recevided your order.</div>
                <div class="headline__content">We've updated your account with your new
                    payment info, as you asked
                </div>
                </div>
                <div class="order">
                <div class="order__title">Your order:</div>
                <div class="order__content">
                    <div>
                    <div class="order__plan">Plan: """+plan_name+"""</div>
                    </div>
                    <div class="order__price">$"""+amount+"""</div>
                </div>
                </div>
                <div class="payment">
                <div class="payment__name">Payment:</div>
                <div class="payment__content">
                    <div class="payment__name">Credit card</div>
                    <div class="payment__caption">
                    <img height="20" src="https://api.eofactory.ai/file/storage/1586917212473_mastercard.png?fbclid=IwAR0KPTP6XHAtV0S4dpYyxaaplrjYPnU0GPqpGCGjILnQ9b-YDCPaMuQ3FTw" alt="mastercard">
                    <span>.... .... .... """+payment_method+"""</span>
                    </div>
                </div>
                </div>
                <hr class="main__line">
                <div class="thank">
                <div class="thank__title">Thank,</div>
                <div class="thank__content">EOFactory team</div>
                </div>
            </div>
            </div>
            <div class="footer">
            <div class="container">
                <span>
                <span>Any query? Get in touch by email or take a look at our</span>
                <a class="footer__link" href="https://app.eofactory.ai/ticket">Support Page</a>
            </span>
            </div>
            </div>
        </body>
    </html>
    """

